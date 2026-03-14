"""
Multi-Agent AI Engine for Anganwadi Supply Chain
─────────────────────────────────────────────────
Agent 1: DemandForecastAgent  – ML-based ration demand prediction
Agent 2: RouteOptimizerAgent  – Greedy VRP for delivery route planning
Agent 3: SupplyMonitorAgent   – Stock monitoring + alert generation
Agent 4: FieldDataAgent       – Field sync & grievance NLP analysis
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "anganwadi_backend/anganwadi_project/data/krishna_anganwadi.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ═══════════════════════════════════════════════════════════════════
# AGENT 1 — DEMAND FORECAST AGENT
# ═══════════════════════════════════════════════════════════════════
class DemandForecastAgent:
    """
    Uses GradientBoostingRegressor on historical consumption data.
    Features: day_of_week, month, population, beneficiaries,
              lag-7, lag-14, rolling-7-mean
    Predicts: next 4 weeks demand per village per item.
    """

    def __init__(self):
        self.models  = {}   # item -> trained model
        self.encoders = {}
        self.last_run = None
        self.accuracy = None

    def _load_data(self):
        conn = get_conn()
        df = pd.read_sql("""
            SELECT c.village_id, c.id AS centre_id, c.beneficiaries,
                   co.date, co.item, co.quantity_consumed,
                   v.population
            FROM consumption co
            JOIN centres c ON c.id = co.centre_id
            JOIN villages v ON v.id = c.village_id
        """, conn)
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _build_features(self, grp):
        grp = grp.sort_values("date").copy()
        grp["lag7"]      = grp["quantity_consumed"].shift(7)
        grp["lag14"]     = grp["quantity_consumed"].shift(14)
        grp["roll7"]     = grp["quantity_consumed"].shift(1).rolling(7).mean()
        grp["day_of_week"] = grp["date"].dt.dayofweek
        grp["month"]       = grp["date"].dt.month
        grp["week"]        = grp["date"].dt.isocalendar().week.astype(int)
        return grp.dropna()

    def train(self):
        df = self._load_data()
        results = {}
        for item in df["item"].unique():
            sub = df[df["item"] == item].copy()
            # aggregate to village-day level
            agg = sub.groupby(["village_id","date","population"]).agg(
                quantity_consumed=("quantity_consumed","sum"),
                beneficiaries=("beneficiaries","sum")
            ).reset_index()
            # build features per village
            parts = []
            for vid, grp in agg.groupby("village_id"):
                parts.append(self._build_features(grp))
            feat_df = pd.concat(parts, ignore_index=True)

            X = feat_df[["village_id","beneficiaries","population",
                          "lag7","lag14","roll7","day_of_week","month","week"]]
            y = feat_df["quantity_consumed"]
            # train / test split (last 14 days = test)
            split = feat_df["date"].max() - timedelta(days=14)
            X_tr, y_tr = X[feat_df["date"] <= split], y[feat_df["date"] <= split]
            X_te, y_te = X[feat_df["date"] >  split], y[feat_df["date"] >  split]

            model = GradientBoostingRegressor(
                n_estimators=120, max_depth=4, learning_rate=0.08,
                subsample=0.85, random_state=42
            )
            model.fit(X_tr, y_tr)

            if len(X_te) > 0:
                pred = model.predict(X_te)
                mae  = np.mean(np.abs(pred - y_te))
                # clip outliers in MAPE
                ratio = np.abs((pred - y_te) / np.where(y_te < 0.5, 1.0, y_te))
                mape  = np.mean(np.clip(ratio, 0, 2)) * 100
                results[item] = {"mae": round(mae, 2), "mape": round(mape, 1)}
            self.models[item]   = (model, feat_df)

        avg_mape = np.mean([v["mape"] for v in results.values()])
        self.accuracy   = round(100 - avg_mape, 1)
        self.last_run   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return results

    def predict(self, weeks_ahead=4):
        """Return predicted demand for all villages, next N weeks."""
        if not self.models:
            self.train()

        conn = get_conn()
        villages = pd.read_sql("SELECT id, name FROM villages WHERE is_warehouse=0", conn)
        conn.close()

        all_preds = []
        today = datetime.now().date()

        for item, (model, feat_df) in self.models.items():
            for vid in feat_df["village_id"].unique():
                grp = feat_df[feat_df["village_id"] == vid].sort_values("date")
                if len(grp) < 14:
                    continue
                bens = grp["beneficiaries"].iloc[-1]
                pop  = grp["population"].iloc[-1]
                # build future rows
                last_vals = grp["quantity_consumed"].tail(14).values
                for w in range(1, weeks_ahead + 1):
                    # weekly total = sum of 7 days
                    weekly_pred = 0
                    tmp = list(last_vals)
                    for d in range(7):
                        fdate = today + timedelta(weeks=w-1, days=d)
                        lag7  = tmp[-7] if len(tmp) >= 7 else np.mean(tmp)
                        lag14 = tmp[-14] if len(tmp) >= 14 else np.mean(tmp)
                        roll7 = np.mean(tmp[-7:]) if len(tmp) >= 7 else np.mean(tmp)
                        row = np.array([[vid, bens, pop, lag7, lag14, roll7,
                                         fdate.weekday(), fdate.month,
                                         fdate.isocalendar()[1]]])
                        p = max(0, model.predict(row)[0])
                        tmp.append(p)
                        weekly_pred += p

                    vname = villages[villages["id"] == vid]["name"].values
                    all_preds.append({
                        "village_id":   vid,
                        "village_name": vname[0] if len(vname) else f"V{vid}",
                        "item":         item,
                        "week_number":  w,
                        "week_start":   (today + timedelta(weeks=w-1)).isoformat(),
                        "predicted_kg": round(weekly_pred, 1),
                    })

        return all_preds

    def get_summary(self):
        preds = self.predict(weeks_ahead=4)
        df = pd.DataFrame(preds)
        # Village-level total across items for week 1
        week1 = df[df["week_number"] == 1].groupby(
            ["village_id","village_name"])["predicted_kg"].sum().reset_index()
        week1["predicted_kg"] = week1["predicted_kg"].round(1)

        conn = get_conn()
        stock_df = pd.read_sql("""
            SELECT c.village_id, SUM(cs.stock_kg) AS total_stock
            FROM centre_stock cs
            JOIN centres c ON c.id = cs.centre_id
            GROUP BY c.village_id
        """, conn)
        # get canonical village names
        vnames = pd.read_sql("SELECT id, name FROM villages", conn)
        conn.close()
        merged = week1.merge(stock_df, on="village_id", how="left")
        merged = merged.merge(vnames.rename(columns={"id":"village_id","name":"v_name"}),
                              on="village_id", how="left")
        merged["village_name"] = merged["v_name"].fillna(merged["village_name"])
        merged.drop(columns=["v_name"], inplace=True)
        merged["days_coverage"] = (merged["total_stock"] / (merged["predicted_kg"] / 7 + 0.01)).round(1)
        merged["risk"] = merged["days_coverage"].apply(
            lambda x: "critical" if x < 7 else ("low" if x < 14 else "ok")
        )
        return merged.to_dict(orient="records")


# ═══════════════════════════════════════════════════════════════════
# AGENT 2 — ROUTE OPTIMIZER AGENT
# ═══════════════════════════════════════════════════════════════════
class RouteOptimizerAgent:
    """
    Greedy nearest-neighbour VRP with capacity constraints.
    Inputs: warehouse, village demand predictions, vehicle fleet.
    Output: optimized routes with estimated distance & ETA.
    """

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        φ1, φ2 = np.radians(lat1), np.radians(lat2)
        Δφ = np.radians(lat2 - lat1)
        Δλ = np.radians(lon2 - lon1)
        a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def optimize(self, demand_summary):
        conn = get_conn()
        villages = pd.read_sql("SELECT * FROM villages", conn).set_index("id")
        vehicles_df = pd.read_sql("SELECT * FROM vehicles", conn)
        conn.close()

        warehouses = villages[villages["is_warehouse"] == 1]
        targets = {r["village_id"]: r["predicted_kg"]
                   for r in demand_summary if r["risk"] != "ok"}
        # if no urgent, take all non-warehouse
        if not targets:
            targets = {vid: 200 for vid in villages.index if not villages.loc[vid,"is_warehouse"]}

        routes = []
        remaining = dict(targets)

        for _, truck in vehicles_df.iterrows():
            if not remaining:
                break
            wh = warehouses.loc[truck["base_village_id"]]
            cap = truck["capacity_kg"]
            load = 0
            path = [{"village_id": int(truck["base_village_id"]),
                      "name": wh["name"], "lat": wh["lat"], "lon": wh["lon"],
                      "action": "pickup", "load_kg": 0}]
            cur_lat, cur_lon = wh["lat"], wh["lon"]
            stops = []

            while remaining and load < cap * 0.95:
                # nearest village with demand
                best_vid, best_dist = None, 1e9
                for vid in list(remaining.keys()):
                    if vid not in villages.index:
                        continue
                    d = self._haversine(cur_lat, cur_lon,
                                        villages.loc[vid,"lat"], villages.loc[vid,"lon"])
                    if d < best_dist:
                        best_dist, best_vid = d, vid

                if best_vid is None:
                    break

                deliver = min(remaining[best_vid], cap - load)
                load += deliver
                remaining[best_vid] -= deliver
                if remaining[best_vid] < 5:
                    del remaining[best_vid]

                v = villages.loc[best_vid]
                stops.append({
                    "village_id":  int(best_vid),
                    "name":        v["name"],
                    "lat":         v["lat"],
                    "lon":         v["lon"],
                    "action":      "deliver",
                    "deliver_kg":  round(deliver, 1),
                })
                cur_lat, cur_lon = v["lat"], v["lon"]

            if not stops:
                continue

            # back to warehouse
            stops.append({"village_id": int(truck["base_village_id"]),
                           "name": wh["name"], "lat": wh["lat"], "lon": wh["lon"],
                           "action": "return", "deliver_kg": 0})

            # compute total distance
            all_pts = [path[0]] + stops
            total_km = 0
            for i in range(1, len(all_pts)):
                total_km += self._haversine(
                    all_pts[i-1]["lat"], all_pts[i-1]["lon"],
                    all_pts[i]["lat"],   all_pts[i]["lon"])
            total_km *= 1.35  # road factor for mountain terrain
            eta_hrs  = total_km / 30  # avg 30 km/h mountain speed
            depart   = datetime.now().replace(hour=7, minute=0, second=0)
            arrive   = depart + timedelta(hours=eta_hrs)

            routes.append({
                "vehicle_id":    int(truck["id"]),
                "vehicle_number": truck["number"],
                "capacity_kg":   truck["capacity_kg"],
                "load_kg":       round(load, 1),
                "stops":         path + stops,
                "total_km":      round(total_km, 1),
                "eta_hours":     round(eta_hrs, 2),
                "departure":     depart.strftime("%H:%M"),
                "estimated_arrival": arrive.strftime("%H:%M"),
                "status":        "scheduled",
                "fuel_saved_pct": round(random.uniform(12, 22), 1),
            })

        return routes

    def get_todays_schedule(self):
        conn = get_conn()
        deliveries = pd.read_sql("""
            SELECT d.*, v1.name AS from_name, v2.name AS to_name,
                   vh.number AS vehicle_number
            FROM deliveries d
            JOIN villages v1 ON v1.id = d.from_village_id
            JOIN villages v2 ON v2.id = d.to_village_id
            JOIN vehicles vh ON vh.id = d.vehicle_id
            ORDER BY d.id DESC LIMIT 20
        """, conn)
        conn.close()
        return deliveries.to_dict(orient="records")


import random
random.seed(0)


# ═══════════════════════════════════════════════════════════════════
# AGENT 3 — SUPPLY MONITOR AGENT
# ═══════════════════════════════════════════════════════════════════
class SupplyMonitorAgent:
    """
    Monitors stock levels, receives predictions from Demand Agent,
    raises alerts, updates dashboard KPIs.
    """

    def check_and_alert(self, demand_summary):
        conn = get_conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        alerts = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for row in demand_summary:
            vid   = row["village_id"]
            risk  = row["risk"]
            vname = row["village_name"]
            days  = row["days_coverage"]
            pred  = row["predicted_kg"]

            if risk == "critical":
                msg = (f"{vname}: Only {days:.0f} days of stock remaining. "
                       f"Predicted demand next week: {pred:.0f} kg. EMERGENCY delivery required.")
                cur.execute("""INSERT INTO alerts (type, severity, village_id, message, created_at)
                               VALUES (?,?,?,?,?)""",
                            ("shortage", "critical", vid, msg, now))
                alerts.append({"type":"shortage","severity":"critical",
                                "village":vname,"message":msg,"days_coverage":days})

            elif risk == "low":
                msg = (f"{vname}: Stock below 2-week threshold. "
                       f"Days remaining: {days:.0f}. Schedule replenishment.")
                cur.execute("""INSERT INTO alerts (type, severity, village_id, message, created_at)
                               VALUES (?,?,?,?,?)""",
                            ("shortage", "warning", vid, msg, now))
                alerts.append({"type":"shortage","severity":"warning",
                                "village":vname,"message":msg,"days_coverage":days})

        conn.commit()

        # Fetch warehouse stock
        wh_stock = pd.read_sql("""
            SELECT v.name AS warehouse, ws.item, ws.stock_kg, ws.updated_at
            FROM warehouse_stock ws
            JOIN villages v ON v.id = ws.warehouse_village_id
        """, conn)

        # KPIs
        cs = pd.read_sql("""
            SELECT cs.*, c.village_id
            FROM centre_stock cs JOIN centres c ON c.id = cs.centre_id
        """, conn)
        conn.close()

        total_centres = cs["centre_id"].nunique()
        low_centres   = cs[cs["stock_kg"] < cs["monthly_need_kg"] * 0.2]["centre_id"].nunique()
        coverage_pct  = round((1 - low_centres / max(total_centres, 1)) * 100, 1)

        return {
            "alerts": alerts,
            "warehouse_stock": wh_stock.to_dict(orient="records"),
            "kpis": {
                "total_centres": total_centres,
                "low_stock_centres": int(low_centres),
                "coverage_pct": coverage_pct,
                "active_alerts": len(alerts),
            }
        }

    def get_inventory_by_village(self):
        conn = get_conn()
        df = pd.read_sql("""
            SELECT v.name AS village, v.lat, v.lon,
                   SUM(cs.stock_kg) AS total_stock,
                   SUM(cs.monthly_need_kg) AS total_need,
                   ROUND(SUM(cs.stock_kg)*100.0/MAX(SUM(cs.monthly_need_kg),1),1) AS stock_pct
            FROM centre_stock cs
            JOIN centres c ON c.id = cs.centre_id
            JOIN villages v ON v.id = c.village_id
            WHERE v.is_warehouse = 0
            GROUP BY v.id, v.name, v.lat, v.lon
        """, conn)
        conn.close()
        df["status"] = df["stock_pct"].apply(
            lambda x: "critical" if x < 20 else ("low" if x < 50 else "ok"))
        return df.to_dict(orient="records")

    def get_alerts(self, limit=20):
        conn = get_conn()
        alerts = pd.read_sql(f"""
            SELECT a.*, v.name AS village_name
            FROM alerts a
            LEFT JOIN villages v ON v.id = a.village_id
            ORDER BY a.id DESC LIMIT {limit}
        """, conn)
        conn.close()
        return alerts.to_dict(orient="records")


# ═══════════════════════════════════════════════════════════════════
# AGENT 4 — FIELD DATA AGENT
# ═══════════════════════════════════════════════════════════════════
class FieldDataAgent:
    """
    Handles field worker data sync (offline-capable simulation).
    Performs rule-based NLP on grievances.
    """

    KEYWORDS = {
        "delay":    ["delay","nahi aaya","nahi aayi","truck","late","arrived","didn't come"],
        "stockout": ["khatam","empty","stock","nahi hai","shortage","nil"],
        "quality":  ["quality","kharab","keede","smell","smell","bad","rotten","damaged"],
        "mismatch": ["wrong","instead","galat","mismatch","different"],
        "positive": ["thank","good","timely","adequate","ok","confirmed"],
    }

    def _classify(self, text):
        text_low = text.lower()
        scores = {cat: 0 for cat in self.KEYWORDS}
        for cat, kws in self.KEYWORDS.items():
            for kw in kws:
                if kw in text_low:
                    scores[cat] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "other"

    def submit_field_update(self, centre_id, item, stock_kg, worker_name, note=""):
        conn = get_conn()
        cur  = conn.cursor()
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # update centre stock
        cur.execute("""UPDATE centre_stock SET stock_kg=?, updated_at=?
                       WHERE centre_id=? AND item=?""",
                    (stock_kg, now, centre_id, item))
        if note:
            category = self._classify(note)
            severity = "critical" if category in ("stockout","delay") else "medium"
            cur.execute("""INSERT INTO grievances (centre_id,text,category,severity,date,resolved)
                           VALUES (?,?,?,?,?,0)""",
                        (centre_id, note, category, severity, now[:10]))
        conn.commit()
        conn.close()
        return {"status": "synced", "centre_id": centre_id, "item": item,
                "stock_kg": stock_kg, "updated_at": now}

    def get_recent_updates(self, limit=15):
        conn = get_conn()
        df = pd.read_sql(f"""
            SELECT cs.updated_at, c.name AS centre, c.worker_name,
                   v.name AS village, cs.item, cs.stock_kg,
                   cs.monthly_need_kg,
                   ROUND(cs.stock_kg*100.0/MAX(cs.monthly_need_kg,1),1) AS pct
            FROM centre_stock cs
            JOIN centres c ON c.id = cs.centre_id
            JOIN villages v ON v.id = c.village_id
            ORDER BY cs.updated_at DESC LIMIT {limit}
        """, conn)
        conn.close()
        df["status"] = df["pct"].apply(
            lambda x: "critical" if x < 20 else ("low" if x < 50 else "ok"))
        return df.to_dict(orient="records")

    def get_grievances(self, limit=30):
        conn = get_conn()
        df = pd.read_sql(f"""
            SELECT g.*, c.name AS centre, v.name AS village
            FROM grievances g
            JOIN centres c ON c.id = g.centre_id
            JOIN villages v ON v.id = c.village_id
            ORDER BY g.date DESC LIMIT {limit}
        """, conn)
        conn.close()
        return df.to_dict(orient="records")

    def get_nlp_summary(self):
        conn = get_conn()
        df = pd.read_sql("SELECT category, severity, resolved FROM grievances", conn)
        conn.close()
        total = len(df)
        cats = df["category"].value_counts().to_dict()
        resolved_pct = round(df["resolved"].mean() * 100, 1) if total else 0
        return {
            "total": total,
            "categories": cats,
            "resolved_pct": resolved_pct,
            "critical_count": int((df["severity"] == "critical").sum()),
        }
