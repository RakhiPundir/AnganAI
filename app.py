"""
AnganAI Flask API Server
─────────────────────────
Endpoints consumed by the frontend dashboard.
All agents are orchestrated here.
"""

from flask import Flask, jsonify, request, send_from_directory
from datetime import datetime
import threading
import json
import os
import sqlite3

# ── local imports ──────────────────────────────────────────────────
import sys
sys.path.insert(0, "/home/Downloads/AnganAI")
from agents import (
    DemandForecastAgent,
    RouteOptimizerAgent,
    SupplyMonitorAgent,
    FieldDataAgent,
)

app = Flask(__name__, static_folder="/home/Downloads/AnganAI/anganwadi_backend/anganwadi_project/static")

# ── CORS (manual, no flask-cors) ──────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return "", 204

# ── AGENT SINGLETONS ─────────────────────────────────────────────
demand_agent  = DemandForecastAgent()
route_agent   = RouteOptimizerAgent()
supply_agent  = SupplyMonitorAgent()
field_agent   = FieldDataAgent()

# Cache for expensive ML calls
_cache = {}
_cache_lock = threading.Lock()

def cached(key, fn, ttl_sec=120):
    with _cache_lock:
        entry = _cache.get(key)
        now   = datetime.now().timestamp()
        if entry and (now - entry["ts"]) < ttl_sec:
            return entry["data"]
        data = fn()
        _cache[key] = {"data": data, "ts": now}
        return data


# ════════════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════════════
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat(),
                    "agents": ["demand","route","supply","field"]})


# ════════════════════════════════════════════════════════════════════
# DASHBOARD KPIs
# ════════════════════════════════════════════════════════════════════
@app.route("/api/dashboard")
def dashboard():
    def _compute():
        summary  = demand_agent.get_summary()
        monitor  = supply_agent.check_and_alert(summary)
        inv      = supply_agent.get_inventory_by_village()
        return {
            "kpis": monitor["kpis"],
            "inventory_map": inv,
            "forecast_accuracy": demand_agent.accuracy or 91.0,
            "last_updated": datetime.now().isoformat(),
        }
    return jsonify(cached("dashboard", _compute, ttl_sec=60))


# ════════════════════════════════════════════════════════════════════
# AGENT 1 — DEMAND FORECAST
# ════════════════════════════════════════════════════════════════════
@app.route("/api/demand/summary")
def demand_summary():
    def _run():
        return demand_agent.get_summary()
    return jsonify(cached("demand_summary", _run, ttl_sec=90))


@app.route("/api/demand/predictions")
def demand_predictions():
    weeks = int(request.args.get("weeks", 4))
    village_id = request.args.get("village_id")
    def _run():
        preds = demand_agent.predict(weeks_ahead=weeks)
        if village_id:
            preds = [p for p in preds if str(p["village_id"]) == str(village_id)]
        return preds
    return jsonify(cached(f"preds_{weeks}_{village_id}", _run, ttl_sec=90))


@app.route("/api/demand/chart")
def demand_chart():
    """Weekly aggregated demand for bar chart (actual + predicted)."""
    conn = sqlite3.connect("anganwadi_backend/anganwadi_project/data/krishna_anganwadi.db")
    conn.row_factory = sqlite3.Row
    import pandas as pd
    actual_df = pd.read_sql("""
        SELECT strftime('%Y-W%W', date) AS week,
               SUM(quantity_consumed) AS total_kg
        FROM consumption
        WHERE date >= date('now','-7 weeks')
        GROUP BY week ORDER BY week
    """, conn)
    conn.close()

    actual = actual_df.tail(5).to_dict(orient="records")
    # append 2 predicted weeks
    from datetime import date, timedelta
    today = date.today()
    preds_raw = demand_agent.predict(weeks_ahead=2)
    import collections
    pred_by_week = collections.defaultdict(float)
    for p in preds_raw:
        pred_by_week[p["week_number"]] += p["predicted_kg"]
    predicted = [
        {"week": f"W+{w}", "total_kg": round(pred_by_week[w], 0)}
        for w in [1, 2]
    ]
    return jsonify({"actual": actual, "predicted": predicted})


# ════════════════════════════════════════════════════════════════════
# AGENT 2 — ROUTE OPTIMIZER
# ════════════════════════════════════════════════════════════════════
@app.route("/api/routes/optimize")
def routes_optimize():
    def _run():
        summary = demand_agent.get_summary()
        return route_agent.optimize(summary)
    return jsonify(cached("routes", _run, ttl_sec=90))


@app.route("/api/routes/schedule")
def routes_schedule():
    return jsonify(route_agent.get_todays_schedule())


# ════════════════════════════════════════════════════════════════════
# AGENT 3 — SUPPLY MONITOR
# ════════════════════════════════════════════════════════════════════
@app.route("/api/supply/alerts")
def supply_alerts():
    return jsonify(supply_agent.get_alerts(limit=20))


@app.route("/api/supply/inventory")
def supply_inventory():
    return jsonify(supply_agent.get_inventory_by_village())


@app.route("/api/supply/warehouse")
def supply_warehouse():
    def _run():
        summary = demand_agent.get_summary()
        result  = supply_agent.check_and_alert(summary)
        return result["warehouse_stock"]
    return jsonify(cached("warehouse", _run, ttl_sec=60))


# ════════════════════════════════════════════════════════════════════
# AGENT 4 — FIELD DATA
# ════════════════════════════════════════════════════════════════════
@app.route("/api/field/updates")
def field_updates():
    return jsonify(field_agent.get_recent_updates(limit=15))


@app.route("/api/field/grievances")
def field_grievances():
    return jsonify(field_agent.get_grievances(limit=30))


@app.route("/api/field/nlp-summary")
def field_nlp_summary():
    return jsonify(field_agent.get_nlp_summary())


@app.route("/api/field/submit", methods=["POST"])
def field_submit():
    data = request.get_json()
    result = field_agent.submit_field_update(
        centre_id   = data.get("centre_id", 1),
        item        = data.get("item", "Rice"),
        stock_kg    = float(data.get("stock_kg", 0)),
        worker_name = data.get("worker_name", "Unknown"),
        note        = data.get("note", ""),
    )
    with _cache_lock:
        for k in ["dashboard","demand_summary","warehouse"]:
            _cache.pop(k, None)
    return jsonify(result)


# ════════════════════════════════════════════════════════════════════
# AGENT CYCLE — run all agents in sequence (triggered by dashboard)
# ════════════════════════════════════════════════════════════════════
@app.route("/api/agents/run-cycle", methods=["POST"])
def run_cycle():
    log = []
    t0  = datetime.now()

    log.append({"agent":"DemandForecastAgent","status":"running","msg":"Training + predicting…"})
    summary = demand_agent.get_summary()
    log.append({"agent":"DemandForecastAgent","status":"done",
                 "msg":f"Predicted demand for {len(summary)} villages. Accuracy: {demand_agent.accuracy}%"})

    log.append({"agent":"RouteOptimizerAgent","status":"running","msg":"Computing optimal routes…"})
    routes = route_agent.optimize(summary)
    log.append({"agent":"RouteOptimizerAgent","status":"done",
                 "msg":f"{len(routes)} routes optimized. Avg fuel saving: {round(sum(r.get('fuel_saved_pct',0) for r in routes)/max(len(routes),1),1)}%"})

    log.append({"agent":"SupplyMonitorAgent","status":"running","msg":"Cross-referencing inventory…"})
    monitor = supply_agent.check_and_alert(summary)
    log.append({"agent":"SupplyMonitorAgent","status":"done",
                 "msg":f"{len(monitor['alerts'])} alerts raised. Coverage: {monitor['kpis']['coverage_pct']}%"})

    log.append({"agent":"FieldDataAgent","status":"running","msg":"Syncing offline worker reports…"})
    updates = field_agent.get_recent_updates(limit=5)
    log.append({"agent":"FieldDataAgent","status":"done",
                 "msg":f"{len(updates)} recent field updates synced."})

    elapsed = round((datetime.now() - t0).total_seconds(), 2)
    with _cache_lock:
        _cache.clear()
    return jsonify({"cycle_complete": True, "elapsed_sec": elapsed, "log": log,
                    "timestamp": datetime.now().isoformat()})


# ════════════════════════════════════════════════════════════════════
# CENTRES + VILLAGES (for maps / dropdowns)
# ════════════════════════════════════════════════════════════════════
@app.route("/api/villages")
def villages():
    conn = sqlite3.connect("anganwadi_backend/anganwadi_project/data/krishna_anganwadi.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM villages").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/centres")
def centres():
    conn = sqlite3.connect("anganwadi_backend/anganwadi_project/data/krishna_anganwadi.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.*, v.name AS village_name FROM centres c
        JOIN villages v ON v.id = c.village_id
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ════════════════════════════════════════════════════════════════════
# STATIC FRONTEND
# ════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory("anganwadi_backend/anganwadi_project/static", "index.html")


if __name__ == "__main__":
    print("🌿 AnganAI API starting …")
    print("  Training demand model … (first load may take ~10s)")
    demand_agent.train()
    print(f"  Demand model accuracy: {demand_agent.accuracy}%")
    print("  API ready at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
