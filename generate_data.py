"""
Dataset Generator — Krishna District, Andhra Pradesh
Anganwadi Supply Chain Optimization

Data Sources:
  - Village names & mandal names : Census of India 2011
      censusindia.co.in, census2011.co.in,
      Wikipedia "List of mandals in Krishna district"
  - Lat/Lon coordinates          : Census 2011 village-level GIS centroids
      (approximated from Google Maps + censusindia.co.in)
  - Population                   : Census of India 2011, Krishna district
      Total: 4,529,009 | Rural: 2,918,732
  - AWC (Anganwadi Centre) count : ICDS scheme norms — 1 AWC per 400–800
      rural population (MoWCD GoI operational guidelines)
  - Ration item quantities       : POSHAN 2.0 / ICDS SNP norms
      Rice 6 kg, Dal 1.5 kg, Oil 0.5 L, Salt 0.5 kg,
      Ragi 1 kg, Eggs 8 nos per beneficiary per month
  - Warehouse locations          : AP Civil Supplies Corporation godowns
      and FCI depots in Krishna district (official AP CS portal)
  - Vehicle registration prefix  : AP-16 (Krishna district RTO code)
  - Worker names                 : Common Telugu female names
  - Grievance text               : Mix of Telugu + English field reports
      modelled on ICDS grievance redressal patterns

Schema: identical to Chamoli prototype — agents.py / app.py unchanged.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

random.seed(16)          # AP-16 = Krishna district RTO code
np.random.seed(16)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "anganwadi_backend/anganwadi_project/data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH  = os.path.join(DATA_DIR, "anganwadi.db")

# ═══════════════════════════════════════════════════════════════════
# VILLAGES — real locations from Census 2011, Krishna district, AP
# is_warehouse=True → AP Civil Supplies / FCI godown at that town
# block = mandal name (equivalent to "block" in other states)
# ═══════════════════════════════════════════════════════════════════
VILLAGES = [
    # ── Machilipatnam Revenue Division ──────────────────────────────
    {"id":  1, "name": "Machilipatnam",  "lat": 16.1875, "lon": 81.1389, "population": 169892, "is_warehouse": True,  "block": "Machilipatnam"},
    {"id":  2, "name": "Pedana",         "lat": 16.2533, "lon": 81.1478, "population":  25140, "is_warehouse": False, "block": "Pedana"},
    {"id":  3, "name": "Bantumilli",     "lat": 16.2561, "lon": 81.2672, "population":  14820, "is_warehouse": False, "block": "Bantumilli"},
    {"id":  4, "name": "Kruthivennu",    "lat": 16.1253, "lon": 81.2094, "population":  12300, "is_warehouse": False, "block": "Kruthivennu"},
    {"id":  5, "name": "Nagayalanka",    "lat": 15.9236, "lon": 80.7614, "population":  17640, "is_warehouse": False, "block": "Nagayalanka"},
    {"id":  6, "name": "Avanigadda",     "lat": 16.0194, "lon": 80.9194, "population":  35120, "is_warehouse": False, "block": "Avanigadda"},
    {"id":  7, "name": "Challapalli",    "lat": 16.1122, "lon": 80.9347, "population":  19870, "is_warehouse": False, "block": "Challapalli"},
    {"id":  8, "name": "Mandavalli",     "lat": 16.3056, "lon": 81.0514, "population":  11450, "is_warehouse": False, "block": "Mandavalli"},
    {"id":  9, "name": "Movva",          "lat": 16.4897, "lon": 81.0497, "population":  16230, "is_warehouse": False, "block": "Movva"},
    {"id": 10, "name": "Kalidindi",      "lat": 16.4781, "lon": 81.2119, "population":  18960, "is_warehouse": False, "block": "Kalidindi"},
    {"id": 11, "name": "Ghantasala",     "lat": 16.1731, "lon": 81.0528, "population":  14300, "is_warehouse": False, "block": "Ghantasala"},
    {"id": 12, "name": "Mopidevi",       "lat": 16.0978, "lon": 80.8736, "population":   9870, "is_warehouse": False, "block": "Mopidevi"},

    # ── Gudivada Revenue Division ────────────────────────────────────
    {"id": 13, "name": "Gudivada",       "lat": 16.4344, "lon": 80.9922, "population": 110756, "is_warehouse": True,  "block": "Gudivada"},
    {"id": 14, "name": "Kankipadu",      "lat": 16.4322, "lon": 80.7631, "population":  28400, "is_warehouse": False, "block": "Kankipadu"},
    {"id": 15, "name": "Nandivada",      "lat": 16.4589, "lon": 80.8997, "population":  21340, "is_warehouse": False, "block": "Nandivada"},
    {"id": 16, "name": "Gudlavalleru",   "lat": 16.3544, "lon": 81.0314, "population":  19870, "is_warehouse": False, "block": "Gudlavalleru"},
    {"id": 17, "name": "Mudinepalli",    "lat": 16.6167, "lon": 81.0619, "population":  15640, "is_warehouse": False, "block": "Mudinepalli"},
    {"id": 18, "name": "Chandarlapadu",  "lat": 16.7789, "lon": 80.8814, "population":  13210, "is_warehouse": False, "block": "Chandarlapadu"},
    {"id": 19, "name": "Pamidimukkala", "lat": 16.3728, "lon": 80.6136, "population":  10980, "is_warehouse": False, "block": "Pamidimukkala"},
    {"id": 20, "name": "Pamarru",        "lat": 16.3197, "lon": 80.9214, "population":  31200, "is_warehouse": False, "block": "Pamarru"},
    {"id": 21, "name": "Vuyyuru",        "lat": 16.3644, "lon": 80.8439, "population":  42300, "is_warehouse": False, "block": "Vuyyuru"},

    # ── Vijayawada / Nandigama Division ──────────────────────────────
    {"id": 22, "name": "Nandigama",      "lat": 16.7722, "lon": 80.2864, "population":  43560, "is_warehouse": True,  "block": "Nandigama"},
    {"id": 23, "name": "Tiruvuru",       "lat": 16.5614, "lon": 80.6011, "population":  37890, "is_warehouse": False, "block": "Tiruvuru"},
    {"id": 24, "name": "Mylavaram",      "lat": 16.7403, "lon": 80.6319, "population":  26710, "is_warehouse": False, "block": "Mylavaram"},
    {"id": 25, "name": "Bapulapadu",     "lat": 16.6792, "lon": 80.4953, "population":  18340, "is_warehouse": False, "block": "Bapulapadu"},
    {"id": 26, "name": "Vissannapet",    "lat": 16.6736, "lon": 80.3519, "population":  15980, "is_warehouse": False, "block": "Vissannapet"},
    {"id": 27, "name": "Gampalagudem",   "lat": 16.8011, "lon": 80.5083, "population":  11230, "is_warehouse": False, "block": "Gampalagudem"},
    {"id": 28, "name": "Kondapalli",     "lat": 16.6178, "lon": 80.5425, "population":   8940, "is_warehouse": False, "block": "Kondapalli"},
    {"id": 29, "name": "Jaggayyapeta",   "lat": 16.8919, "lon": 80.0989, "population":  57430, "is_warehouse": False, "block": "Jaggayyapeta"},
    {"id": 30, "name": "Vatsavai",       "lat": 16.6972, "lon": 80.7703, "population":  13780, "is_warehouse": False, "block": "Vatsavai"},
]

# ═══════════════════════════════════════════════════════════════════
# CENTRES — generated from VILLAGES (1 AWC per ~600 rural pop)
# Worker names are common Telugu female names (ICDS workers)
# ═══════════════════════════════════════════════════════════════════
TELUGU_WORKERS = [
    "Lakshmi Devi","Sarojini","Padmavathi","Subbulakshmi","Rajeswari","Annapurna",
    "Kalavathi","Sumitra Devi","Savithri","Nageswari","Vimala","Kamakshi","Bhavani",
    "Swarajyalakshmi","Rathnamma","Vijayalakshmi","Leelavathi","Seshamma",
    "Krishnaveni","Nagamani","Varalakshmi","Indira","Tulasi","Sarala","Hymavathi",
    "Bhagyalakshmi","Nirmala","Chandrakala","Pushpa","Lalitha","Usha Rani",
    "Meenakshi","Alamelu","Revathi","Santha","Gowri","Manjula","Sunanda",
    "Durga Devi","Ammaji","Jyothi","Rekha","Aruna","Kumari","Hema","Radha",
]

CENTRES = []
cid = 1
for v in VILLAGES:
    # warehouses also have centres in their town
    n_centres = max(1, v["population"] // 600)
    for j in range(n_centres):
        CENTRES.append({
            "id": cid,
            "village_id":    v["id"],
            "name":          f"{v['name']} AWC-{j+1:02d}",
            "worker_name":   random.choice(TELUGU_WORKERS),
            "beneficiaries": random.randint(35, 95),
            "has_connectivity": random.random() > 0.15,   # AP better connectivity than Uttarakhand
        })
        cid += 1

# ═══════════════════════════════════════════════════════════════════
# VEHICLES — AP-16 = Krishna district RTO prefix
# Based at warehouse villages (is_warehouse=True)
# ═══════════════════════════════════════════════════════════════════
VEHICLES = [
    # Machilipatnam godown fleet
    {"id": 1, "number": "AP-16-TC-3401", "capacity_kg": 700, "base_village_id":  1},
    {"id": 2, "number": "AP-16-TC-3402", "capacity_kg": 600, "base_village_id":  1},
    {"id": 3, "number": "AP-16-TC-3403", "capacity_kg": 800, "base_village_id":  1},
    # Gudivada godown fleet
    {"id": 4, "number": "AP-16-TC-5801", "capacity_kg": 650, "base_village_id": 13},
    {"id": 5, "number": "AP-16-TC-5802", "capacity_kg": 700, "base_village_id": 13},
    {"id": 6, "number": "AP-16-TC-5803", "capacity_kg": 500, "base_village_id": 13},
    # Nandigama sub-depot fleet
    {"id": 7, "number": "AP-16-TC-7201", "capacity_kg": 600, "base_village_id": 22},
    {"id": 8, "number": "AP-16-TC-7202", "capacity_kg": 550, "base_village_id": 22},
]

# ═══════════════════════════════════════════════════════════════════
# RATION ITEMS — ICDS / POSHAN 2.0 norms for AP
# Jaggery replaced by Ragi (more common in AP ICDS supplementary nutrition)
# ═══════════════════════════════════════════════════════════════════
RATION_ITEMS = ["Rice", "Red Gram Dal", "Groundnut Oil", "Salt", "Ragi", "Wheat", "Eggs"]
ITEM_UNIT    = {
    "Rice": "kg", "Red Gram Dal": "kg", "Groundnut Oil": "ltr",
    "Salt": "kg", "Ragi": "kg", "Wheat": "kg", "Eggs": "count"
}
ITEM_PER_BEN = {   # per beneficiary per month (POSHAN 2.0 norms)
    "Rice": 6.0, "Red Gram Dal": 1.5, "Groundnut Oil": 0.5,
    "Salt": 0.5, "Ragi": 1.0, "Wheat": 2.0, "Eggs": 8.0
}

# ═══════════════════════════════════════════════════════════════════
# GENERATORS
# ═══════════════════════════════════════════════════════════════════

def generate_consumption(days=90):
    """Daily consumption per centre per item — 90 days history."""
    records = []
    start = datetime.now() - timedelta(days=days)
    for centre in CENTRES:
        bens = centre["beneficiaries"]
        for item in RATION_ITEMS:
            base = ITEM_PER_BEN[item] * bens / 30   # daily average
            for d in range(days):
                dt = start + timedelta(days=d)
                month = dt.month
                # AP seasonal: higher Jun–Oct (Kharif + monsoon surge)
                seasonal = 1 + 0.12 * np.sin(2 * np.pi * (month - 4) / 12)
                noise = np.random.normal(1.0, 0.10)
                # ~8% zero days (Sundays, public holidays, school holidays)
                qty = 0.0 if random.random() < 0.08 else round(max(0, base * seasonal * noise), 2)
                records.append({
                    "centre_id":         centre["id"],
                    "date":              dt.strftime("%Y-%m-%d"),
                    "item":              item,
                    "quantity_consumed": qty,
                })
    return pd.DataFrame(records)


def generate_inventory():
    """Current warehouse + centre stock levels."""
    wh_records = []
    for v in VILLAGES:
        if v["is_warehouse"]:
            for item in RATION_ITEMS:
                wh_records.append({
                    "warehouse_village_id": v["id"],
                    "item":       item,
                    "stock_kg":   round(random.uniform(3000, 12000), 1),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

    centre_records = []
    for c in CENTRES:
        for item in RATION_ITEMS:
            monthly_need = ITEM_PER_BEN[item] * c["beneficiaries"]
            frac = random.choices(
                [0.03, 0.12, 0.30, 0.60, 0.90, 1.10],
                weights=[0.05, 0.10, 0.18, 0.30, 0.25, 0.12]
            )[0]
            centre_records.append({
                "centre_id":       c["id"],
                "item":            item,
                "stock_kg":        round(monthly_need * frac, 1),
                "monthly_need_kg": round(monthly_need, 1),
                "updated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
    return pd.DataFrame(wh_records), pd.DataFrame(centre_records)


def generate_deliveries(n=300):
    """Historical delivery records."""
    warehouses = [v for v in VILLAGES if v["is_warehouse"]]
    non_wh     = [v for v in VILLAGES if not v["is_warehouse"]]
    statuses   = ["delivered", "delivered", "delivered", "delayed", "partial"]
    start      = datetime.now() - timedelta(days=90)
    records    = []
    for i in range(n):
        v_from    = random.choice(warehouses)
        v_to      = random.choice(non_wh)
        truck     = random.choice(VEHICLES)
        d         = start + timedelta(days=random.randint(0, 89))
        status    = random.choices(statuses, weights=[0.65, 0.65, 0.65, 0.15, 0.10])[0]
        delay_hrs = random.randint(1, 6) if status == "delayed" else 0
        records.append({
            "id":             i + 1,
            "vehicle_id":     truck["id"],
            "from_village_id": v_from["id"],
            "to_village_id":   v_to["id"],
            "item":            random.choice(RATION_ITEMS),
            "quantity_kg":     round(random.uniform(60, truck["capacity_kg"] * 0.85), 1),
            "scheduled_date":  d.strftime("%Y-%m-%d"),
            "status":          status,
            "delay_hours":     delay_hrs,
            "created_at":      (d - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(records)


def generate_grievances(n=120):
    """
    Field grievance reports — Telugu + English mix
    reflecting real AP Anganwadi complaint patterns.
    """
    templates = [
        # Telugu language complaints (transliterated)
        ("Truck iddu ravaledu, pillalu empty stomach lo vellipoyyaru.",       "delay",    "critical"),
        ("Biyyam quality chala daridragaa undi, raallalu kanapadutunnaayi.",   "quality",  "high"),
        ("Stock aipoyindi, department nundi emee update raaledhu.",            "stockout", "critical"),
        ("Groundnut oil quantity takkuva ichharu, sanction kante teesukupoyyaru.", "mismatch","high"),
        ("Vegus delivery chese driver meeda complain cheyyali, mee illu kaadu.", "delay",  "high"),
        ("Gurtanchi kallu kuda raaledu idhe naela, pillalaki chestha ento.",   "delay",   "critical"),
        # English language complaints
        ("Truck did not arrive this week. Children sent home without food.",   "delay",    "critical"),
        ("Rice quality was very poor, had stones and smelled bad.",            "quality",  "high"),
        ("Stock exhausted 12 days ago. No update from mandal office.",         "stockout", "critical"),
        ("Driver demanded bribe to deliver to our village.",                   "corruption","critical"),
        ("Delivery was 4 days late due to flooded canal road.",                "delay",    "high"),
        ("Eggs were cracked and unusable — nearly 40% damaged in transport.",  "quality",  "medium"),
        ("Received wheat flour instead of rice this month.",                   "mismatch", "medium"),
        ("Salt stock exhausted. Children not receiving proper nutrition.",      "stockout", "high"),
        ("AWC building flooded during rains, stock damaged.",                  "infrastructure","high"),
        ("Worker registration count is wrong — 12 beneficiaries missing.",    "mismatch", "medium"),
        ("Delivery on time this week. Worker appreciated the effort.",         "positive", "low"),
        ("Stock adequate and of good quality. No issues.",                     "positive", "low"),
        ("Ragi delivered was damp and had fungus smell.",                      "quality",  "high"),
        ("Mandal supervisor not visiting for last 2 months.",                  "oversight","medium"),
    ]
    records = []
    start   = datetime.now() - timedelta(days=30)
    for i in range(n):
        tmpl     = random.choice(templates)
        c        = random.choice(CENTRES)
        d        = start + timedelta(days=random.randint(0, 29))
        resolved = random.random() > 0.35
        records.append({
            "id":        i + 1,
            "centre_id": c["id"],
            "text":      tmpl[0],
            "category":  tmpl[1],
            "severity":  tmpl[2],
            "date":      d.strftime("%Y-%m-%d"),
            "resolved":  int(resolved),
        })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════
# BUILD DATABASE
# ═══════════════════════════════════════════════════════════════════
def build_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Villages
    cur.execute("""CREATE TABLE villages (
        id INTEGER PRIMARY KEY, name TEXT, lat REAL, lon REAL,
        population INTEGER, is_warehouse INTEGER, block TEXT)""")
    for v in VILLAGES:
        cur.execute("INSERT INTO villages VALUES (?,?,?,?,?,?,?)",
            (v["id"], v["name"], v["lat"], v["lon"],
             v["population"], int(v["is_warehouse"]), v["block"]))
    print(f"  ✓ villages: {len(VILLAGES)}")

    # Centres
    cur.execute("""CREATE TABLE centres (
        id INTEGER PRIMARY KEY, village_id INTEGER, name TEXT,
        worker_name TEXT, beneficiaries INTEGER, has_connectivity INTEGER)""")
    for c in CENTRES:
        cur.execute("INSERT INTO centres VALUES (?,?,?,?,?,?)",
            (c["id"], c["village_id"], c["name"],
             c["worker_name"], c["beneficiaries"], int(c["has_connectivity"])))
    print(f"  ✓ centres: {len(CENTRES)}")

    # Vehicles
    cur.execute("""CREATE TABLE vehicles (
        id INTEGER PRIMARY KEY, number TEXT, capacity_kg REAL, base_village_id INTEGER)""")
    for v in VEHICLES:
        cur.execute("INSERT INTO vehicles VALUES (?,?,?,?)",
            (v["id"], v["number"], v["capacity_kg"], v["base_village_id"]))
    print(f"  ✓ vehicles: {len(VEHICLES)}")

    conn.commit()

    # Consumption
    print(f"Generating consumption (90 days × {len(CENTRES)} centres × {len(RATION_ITEMS)} items)…")
    cons_df = generate_consumption(days=90)
    cons_df.to_sql("consumption", conn, if_exists="replace", index=False)
    print(f"  ✓ consumption: {len(cons_df):,} records")

    # Inventory
    wh_df, centre_df = generate_inventory()
    wh_df.to_sql("warehouse_stock", conn, if_exists="replace", index=False)
    centre_df.to_sql("centre_stock",    conn, if_exists="replace", index=False)
    print(f"  ✓ warehouse_stock: {len(wh_df)} rows")
    print(f"  ✓ centre_stock:    {len(centre_df):,} rows")

    # Deliveries
    del_df = generate_deliveries(300)
    del_df.to_sql("deliveries", conn, if_exists="replace", index=False)
    print(f"  ✓ deliveries: {len(del_df)}")

    # Grievances
    gr_df = generate_grievances(120)
    gr_df.to_sql("grievances", conn, if_exists="replace", index=False)
    print(f"  ✓ grievances: {len(gr_df)}")

    # Alerts (written by Supply Monitor Agent at runtime)
    cur.execute("""CREATE TABLE alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, severity TEXT, village_id INTEGER,
        centre_id INTEGER, message TEXT,
        created_at TEXT, resolved INTEGER DEFAULT 0)""")

    conn.commit()
    conn.close()

    size_mb = os.path.getsize(DB_PATH) / 1024 / 1024
    print(f"\n✅ Krishna District DB ready: {DB_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("🌿 AnganAI — Krishna District, Andhra Pradesh")
    print("=" * 50)
    build_db()
