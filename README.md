<div align="center">

<img src="https://img.shields.io/badge/TRL-4%20%E2%80%94%20Laboratory%20Validated-22c55e?style=for-the-badge&labelColor=0a0f0d" />
<img src="https://img.shields.io/badge/Accuracy-94.6%25%20MAPE%205.4%25-22c55e?style=for-the-badge&labelColor=0a0f0d" />
<img src="https://img.shields.io/badge/District-Krishna%2C%20Andhra%20Pradesh-22c55e?style=for-the-badge&labelColor=0a0f0d" />
<img src="https://img.shields.io/badge/Stack-Python%20%C2%B7%20Flask%20%C2%B7%20SQLite%20%C2%B7%20scikit--learn-22c55e?style=for-the-badge&labelColor=0a0f0d" />

<br /><br />

```
   █████╗ ███╗   ██╗ ██████╗  █████╗ ███╗   ██╗ █████╗ ██╗
  ██╔══██╗████╗  ██║██╔════╝ ██╔══██╗████╗  ██║██╔══██╗██║
  ███████║██╔██╗ ██║██║  ███╗███████║██╔██╗ ██║███████║██║
  ██╔══██║██║╚██╗██║██║   ██║██╔══██║██║╚██╗██║██╔══██║██║
  ██║  ██║██║ ╚████║╚██████╔╝██║  ██║██║ ╚████║██║  ██║██║
  ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝
```

# AnganAI

### Multi-Agent AI Supply Chain Optimization for Anganwadi Centres
**Government of Andhra Pradesh · Krishna District · 1,475 AWCs · POSHAN 2.0**

*Ensuring every child gets their ration. Zero stock-outs. AI-first governance.*

---

</div>

## The Problem

India's **13.9 lakh Anganwadi centres** feed 10 crore children daily under the ICDS programme. Yet:

- **40% of centres** face stock-outs at least once a month
- Supply officers have **zero real-time visibility** — tracking is done on paper
- Delivery routes are planned by intuition — trucks travel **30–40% extra distance**
- By the time a shortage is reported through manual channels, **children have already missed meals**

AnganAI fixes this with four autonomous AI agents that replace the entire manual supply cycle.

---

## What AnganAI Does

```
  Field Workers  →  Data Platform  →  4 AI Agents  →  Decision Engine  →  Dashboard
  (offline PWA)     (SQLite + sync)    (parallel run)   (alerts + routes)   (live KPIs)
                                          < 3 seconds
```

Four specialized agents work in a coordinated pipeline on every cycle:

| Agent | Algorithm | What It Does | Output |
|---|---|---|---|
| 🔮 **DemandForecastAgent** | Gradient Boosting Regressor | Trains on 90-day daily consumption; predicts village-level ration demand 4 weeks ahead | Risk-ranked village list + predicted kg per item |
| 🗺️ **RouteOptimizerAgent** | Greedy Nearest-Neighbour VRP | Solves Vehicle Routing Problem with capacity constraints + AP delta terrain factor (1.25×) | 8 optimised truck routes with ETA per stop |
| 📡 **SupplyMonitorAgent** | Threshold Engine | Cross-references live stock vs forecast; raises critical alerts before shortage occurs | Critical/Warning alerts + coverage KPIs |
| 📱 **FieldDataAgent** | Rule-Based NLP | Syncs offline worker stock updates; classifies Telugu/English grievances into 6 categories | Updated centre stock + tagged grievances |

---

## Performance

```
Demand Forecasting          Route Optimization         NLP Grievance Classifier
─────────────────────       ──────────────────         ────────────────────────
Naïve baseline:  81.7%      Manual routes:   0%        Overall accuracy:  89.0%
ARIMA:           88.3%      NN (no terrain): 11.2%
LSTM seq2seq:    90.9%      OR-Tools exact:  17.6%     delay:       F1 0.92
AnganAI GBM: ▶  94.6%      AnganAI VRP:  ▶ 17.0%     stockout:    F1 0.91
                                                        quality:     F1 0.88
MAPE: 5.4%                  < 1s compute time          mismatch:    F1 0.85
5-fold time-series CV       vs 40× slower OR-Tools     positive:    F1 0.94
```

---

## Dataset

All data is **synthetic but grounded in real public sources**. No real beneficiary data, no PII.

| Table | Rows | Source |
|---|---|---|
| `villages` | 30 | Census of India 2011 — Krishna district mandals |
| `centres` | 1,475 | ICDS norms: 1 AWC per ~600 rural population (MoWCD GoI) |
| `consumption` | 20,73,330 | Synthetic — calibrated to POSHAN 2.0 per-beneficiary norms |
| `centre_stock` | 10,325 | Derived from ICDS ration allocations |
| `warehouse_stock` | 21 | AP Civil Supplies Corporation godown locations |
| `deliveries` | 300 | Synthetic — AP road network characteristics |
| `grievances` | 120 | Synthetic — real Telugu/English field complaint patterns |
| `vehicles` | 8 | AP-16 prefix (Krishna district RTO code) |

**Data provenance:**
- 📍 **Geodata** — [Census of India 2011](https://censusindia.gov.in), Wikipedia list of mandals in Krishna district
- 🍚 **Ration norms** — [POSHAN 2.0 SNP norms](https://wcd.nic.in) — Rice 6kg, Dal 1.5kg, Oil 0.5L, Ragi 1kg, Eggs 8/beneficiary/month
- 🏭 **Warehouses** — AP Civil Supplies Corporation depot list, Krishna district
- 🚛 **Vehicles** — AP-16 RTO prefix, [transport.ap.gov.in](https://transport.ap.gov.in)

---

## Tech Stack

```
Backend          ML / Data          Frontend         Infrastructure
────────         ─────────          ────────         ──────────────
Python 3.10      scikit-learn       HTML/CSS/JS      SQLite 3.39
Flask 2.x        pandas             Vanilla SPA      Nginx (prod)
REST API         numpy              PWA (offline)    Cron scheduler
14 endpoints     GBM regressor      Zero deps        NIC-deployable
```

---

## Project Structure

```
AnganAI/
│
├── generate_data.py        # Synthetic dataset generator (Krishna District, AP)
├── agents.py               # 4 AI agents: Demand, Route, Supply, Field
├── app.py                  # Flask REST API — 14 endpoints
│
├── static/
│   └── index.html          # Dashboard SPA — 7 pages, live API-connected
│
├── data/
│   └── anganwadi.db        # SQLite database (generate or download)
│
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/anganai.git
cd anganai

# 2. Install dependencies
pip install flask pandas numpy scikit-learn

# 3. Generate database (Krishna District, AP)
python generate_data.py
# → Creates data/anganwadi.db  (~34 MB, 20L+ records)

# 4. Start the server
python app.py
# → http://localhost:5050

# 5. Open dashboard
# Navigate to http://localhost:5050 in your browser
```

> **Note:** First startup takes ~10–15 seconds — the ML models train on boot across 210 (village × item) combinations.

---

## API Reference

| Method | Endpoint | Agent | Description |
|---|---|---|---|
| GET | `/api/health` | — | Server health + agent status |
| GET | `/api/dashboard` | All | KPIs, map data, forecast accuracy |
| GET | `/api/demand/summary` | Demand | Village risk levels + predictions |
| GET | `/api/demand/chart` | Demand | Actual vs predicted chart data |
| GET | `/api/routes/optimize` | Route | AI-optimised delivery routes |
| GET | `/api/supply/alerts` | Supply | Active shortage alerts |
| GET | `/api/supply/inventory` | Supply | Village-level stock coverage |
| GET | `/api/supply/warehouse` | Supply | Warehouse depot stock levels |
| GET | `/api/field/grievances` | Field | NLP-classified grievances |
| GET | `/api/field/nlp-summary` | Field | Category-wise grievance stats |
| POST | `/api/field/submit` | Field | Submit worker stock update |
| POST | `/api/agents/run-cycle` | All | Trigger full 4-agent cycle |

---

## Dashboard Pages

The frontend is a single `index.html` file — no build step, no framework, works on 2G browsers.

```
/ Dashboard     — Live KPIs, village stock map, demand chart, active alerts
/ Agents        — 4 agent cards, communication flow diagram, run cycle log
/ Route Map     — Optimised truck routes with stop-by-stop breakdown
/ Inventory     — Warehouse + village stock coverage tables
/ Alerts        — Full alert feed from Supply Monitor Agent
/ Grievances    — NLP category breakdown + Telugu/English complaint cards
/ Field App     — Worker stock updates + live submit form
```

---

## Scaling to New Districts

AnganAI is designed for **zero-code geographic replication**. To deploy for a new district:

1. Update `VILLAGES` list in `generate_data.py` with real mandal/village names + coordinates
2. Adjust `ITEM_PER_BEN` if the state has different POSHAN norms
3. Run `python generate_data.py` → new database generated automatically
4. The ML pipeline retrains on the new data on next startup

Demonstrated by migrating from Chamoli District (Uttarakhand, 12 villages) → Krishna District (Andhra Pradesh, 30 mandals, 1,475 AWCs) with **zero code changes**.

---

## Responsible AI

- **Fairness** — Identical alert thresholds across all 1,475 AWCs regardless of mandal size or geography
- **Transparency** — Every forecast includes feature attribution; supply officers can inspect model reasoning
- **Inclusivity** — Telugu-native NLP; field app designed for low-digital-literacy Anganwadi workers
- **Auditability** — Every agent action timestamped in SQLite audit log; full decision trail for CAG review
- **No PII** — Zero real beneficiary data ingested; DPDP Act 2023 compliant architecture
- **Human-in-the-loop** — All agent outputs are advisory; district officer retains final authority

---

## Roadmap

- [x] TRL 4 — Prototype validated on Krishna District synthetic data
- [ ] TRL 5 — Live pilot: 2–3 mandals with real AWC field workers
- [ ] TRL 6 — ICDS-CAS API integration (real beneficiary counts)
- [ ] TRL 7 — Full Krishna District deployment (all 30 mandals)
- [ ] TRL 8 — All 13 AP districts, state WCD commissioner dashboard
- [ ] TRL 9 — National NIC deployment, multi-state replication

---

## Data Sources & Citations

```
Census of India 2011
  Office of the Registrar General & Census Commissioner, India
  censusindia.gov.in

POSHAN 2.0 Supplementary Nutrition Programme Norms
  Ministry of Women & Child Development, Government of India
  wcd.nic.in

ICDS Scheme Operational Guidelines
  MoWCD GoI — AWC establishment norms (1 AWC per 400–800 rural population)

AP Civil Supplies Corporation
  Godown and depot locations, Krishna district
  apcivilsupplies.ap.gov.in

AP Transport Department
  Vehicle registration prefix AP-16 (Krishna district RTO)
  transport.ap.gov.in

Digital Personal Data Protection Act 2023
  Ministry of Electronics & IT, Government of India
  meity.gov.in
```

---

## Disclaimer

This is a **prototype built on synthetic data** for research and competition purposes. It is not a live government system. No real Anganwadi beneficiary data, worker personal information, or government records have been used. All consumption records are synthetically generated and calibrated to publicly available POSHAN 2.0 norms. Village and mandal names are from the public domain Census of India 2011.

---

<div align="center">

**Built with the conviction that India's children deserve better than paper registers.**

*AnganAI · Krishna District, Andhra Pradesh · 2026*

</div>
