# EduSight — ML & Quantum Backend

> **Data & ML Developer deliverables**

> Dropout risk prediction, What-If simulation, quantum-optimised intervention planning, and cloud API layer for the EduSight student retention system.

---

## File Overview

```
edusight/
├── risk_engine.py           # Rule-based risk scoring + What-If simulation
├── recommendation_engine.py # Intervention recommendation engine
├── ml_engine.py             # Classical ML (Random Forest + Gradient Boosting)
├── quantum_engine.py        # Quantum VQC classifier + intervention optimizer
├── cloud_api.py             # FastAPI server — frontend ↔ cloud ML bridge
├── test_edusight.py         # Full integration test suite
├── .env.example             # Environment variable template
├── Maths.csv                # UCI student performance dataset (Math)
└── Portuguese.csv           # UCI student performance dataset (Portuguese) [optional]
```

---

## Architecture

```
Vue.js Frontend (Person 1 & 2)
        │  HTTP JSON
        ▼
┌─────────────────────────────────────────────┐
│              cloud_api.py                   │
│  FastAPI — 8 REST endpoints                 │
│  • Routes requests to the right engine      │
│  • Cloud-first ML (falls back to local)     │
│  • CORS, request logging, async I/O         │
└────┬──────────────┬───────────────┬─────────┘
     │              │               │
     ▼              ▼               ▼
risk_engine   ml_engine       quantum_engine
(rule-based   (Random Forest  (VQC classifier +
 + blending)   + GB + LR)      QIO optimizer)
     │
     ▼
recommendation_engine
(intervention cards)
```

**Score blending formula** (when all engines are active):

```
final_score = 50% × rule_based  +  30% × ML_score  +  20% × quantum_score
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10 or newer
- Your `Maths.csv` (and optionally `Portuguese.csv`) in the project folder

### 2. Install dependencies

Create and activate a virtual environment (recommended):
python -m venv venv

# Windows (Git Bash)
source venv/Scripts/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

Install all required packages:
pip install -r requirements.txt

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if you have a cloud ML service URL — leave blank for local-only mode
```

---

## Running the Engines

### Step 1 — Train the classical ML model

```bash
python ml_engine.py Maths.csv
# Output: edusight_ml_model.joblib
```

With both datasets (recommended — more training data):
```bash
python ml_engine.py Maths.csv Portuguese.csv
```

### Step 2 — Train the Quantum Risk Classifier

```bash
python quantum_engine.py Maths.csv
# Output: edusight_qrc_weights.npy  +  edusight_qrc_weights_norm_min.npy/max.npy
```

> ⏱ The quantum circuit takes ~2–5 minutes to train (40 epochs on CPU).
> Skip this step if you only need the rule-based + classical ML pipeline.

### Step 3 — Start the API server

```bash
uvicorn cloud_api:app --reload --port 8000
```

The server starts at **http://localhost:8000**

Interactive API docs (Swagger UI): **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint | Description | Called by |
|--------|----------|-------------|-----------|
| `GET` | `/health` | Service health check | Person 5 CI/CD, Vue startup |
| `GET` | `/students` | List all students | Person 1 Sidebar.vue |
| `GET` | `/students/{id}` | Single student profile | Person 1 StudentProfile.vue |
| `POST` | `/simulate` | **What-If simulation** ★ | Person 2 sliders |
| `POST` | `/simulate/quantum` | Quantum-optimised plan | Person 2 Quantum button |
| `POST` | `/ml/predict` | Raw ML risk prediction | Person 3 backend |
| `GET` | `/ml/model/status` | Model availability check | Person 5 deployment |
| `POST` | `/ml/model/train` | Trigger model retraining | Person 5 CI/CD pipeline |
| `GET` | `/recommendations/{id}` | Baseline recommendations | Person 2 RecommendationCard.vue |

### Example: POST /simulate

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "student": {
      "student_id": "STU-001",
      "name": "Muhammad Ali bin Faisal",
      "grade": "Form 4",
      "attendance_rate": 62.0,
      "academic_score": 49.0,
      "socio_score": 55.0,
      "family_support": 40.0,
      "trend": "worsening"
    },
    "attendance_boost": 18.0,
    "academic_boost": 10.0,
    "counselling_sessions": 5,
    "welfare_support": 1.0,
    "use_ml": true,
    "use_quantum": false
  }'
```

**Response structure:**
```json
{
  "student_id": "STU-001",
  "risk": {
    "rule_based_score": 46.8,
    "ml_score": 71.2,
    "quantum_score": null,
    "final_score": 57.4,
    "risk_level": "medium",
    "factor_weights": { "attendance": 45.2, "academic_performance": 31.1, ... },
    "dropout_prob_3m": 0.712,
    "ml_source": "local"
  },
  "simulation": {
    "baseline_score": 57.4,
    "projected_score": 32.1,
    "score_delta": -25.3,
    "projected_prob_3m": 0.341,
    "summary": "With attendance improvement, academic tutoring..."
  },
  "recommendations": [
    {
      "urgency": "critical",
      "title": "Immediate parent/guardian contact",
      "description": "Increase attendance to at least 80%...",
      "owner": "Homeroom Teacher"
    }
  ],
  "narrative": "At the ongoing trajectory, Muhammad Ali..."
}
```

### Example: POST /simulate/quantum

```bash
curl -X POST http://localhost:8000/simulate/quantum \
  -H "Content-Type: application/json" \
  -d '{
    "student": {
      "student_id": "STU-001",
      "name": "Muhammad Ali bin Faisal",
      "grade": "Form 4",
      "attendance_rate": 62.0,
      "academic_score": 49.0,
      "socio_score": 55.0,
      "family_support": 40.0,
      "trend": "worsening"
    },
    "budget": 40.0,
    "n_steps": 30
  }'
```

---

## Running Tests

```bash
# TEST A — Rule-based engine only (instant, no CSV needed)
python test_edusight.py rule

# TEST B — Recommendation engine
python test_edusight.py rule   # recommendations are tested inside 'rule'

# TEST C — Classical ML pipeline (requires Maths.csv)
python test_edusight.py ml Maths.csv

# TEST D — Cloud API endpoints (no server needed — uses TestClient)
python test_edusight.py api

# TEST E — Quantum engine (QIO runs without CSV; QRC requires CSV)
python test_edusight.py quantum
python test_edusight.py quantum Maths.csv   # full quantum including VQC training

# TEST F — Everything
python test_edusight.py all Maths.csv
```

### Expected output (all tests):

```
╔══════════════════════════════════════════════════╗
║       EduSight — Integration Test Runner        ║
╚══════════════════════════════════════════════════╝

[PASS] risk_engine — score returns a value
[PASS] risk_engine — high-risk student classified HIGH or MEDIUM
...
[PASS] cloud_api — POST /simulate returns 200
[PASS] cloud_api — /simulate projected < baseline
...
  ✓ All 57 tests passed.
```

---

## Cloud ML Service Integration

EduSight uses a **cloud-first, local-fallback** pattern for ML inference.

### How it works

```
POST /simulate
      │
      ├─ use_ml=true AND ml_data present?
      │         │
      │         ├─ EDUSIGHT_ML_SERVICE_URL set?
      │         │         │
      │         │         ├─ YES → forward to cloud ML service (httpx async)
      │         │         │         └─ timeout / error → fall back to local
      │         │         │
      │         │         └─ NO  → use local MLEngine directly
      │         │
      │         └─ ml_data missing → skip ML, use rule-based only
      │
      └─ Always runs rule-based RiskEngine (never skipped)
```

### Deploying the ML service to cloud

The cloud ML service should expose the same API contract:

```
POST {ML_SERVICE_URL}/predict
Headers: X-API-Key: {your-key}
Body:    { "student_id": "...", "student_data": { ...csv columns... } }
Response:{ "ml_risk_score": 75.2, "dropout_probability": 0.71, ... }

GET  {ML_SERVICE_URL}/health
Response:{ "status": "ok" }

POST {ML_SERVICE_URL}/train
Body:    { "csv_paths": [...], "n_epochs": 40 }
Response:{ "job_id": "...", "status": "started" }
```

**Recommended cloud platforms:**
- **Azure**: Deploy as Azure Container App or Azure Functions
- **GCP**: Cloud Run (containerise with Docker — see below)
- **AWS**: Lambda + API Gateway or ECS Fargate

### Docker (for Person 5)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py .env ./
COPY Maths.csv ./
RUN python ml_engine.py Maths.csv       # pre-train at build time
EXPOSE 8000
CMD ["uvicorn", "cloud_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t edusight-api .
docker run -p 8000:8000 --env-file .env edusight-api
```

---

## Integration Guide for Other Team Members

### Person 1 — Frontend Lead

Connect the Vue router to the API:

```javascript
// In your Vue service file (e.g. api.js)
const API_BASE = 'http://localhost:8000'

export const getStudent = (id) =>
  fetch(`${API_BASE}/students/${id}`).then(r => r.json())

export const runSimulation = (payload) =>
  fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(r => r.json())
```

### Person 2 — Frontend Components Developer

Slider onChange handler (AttendanceSlider.vue / GradeSlider.vue):

```javascript
// Call POST /simulate every time a slider changes
async onSliderChange() {
  const response = await runSimulation({
    student: this.student,
    attendance_boost: this.attendanceBoost,
    academic_boost: this.academicBoost,
    counselling_sessions: this.counselSessions,
    welfare_support: this.welfareSupport,
    use_ml: true,
    use_quantum: false,
  })
  this.simulationResult  = response.simulation
  this.riskData          = response.risk
  this.recommendations   = response.recommendations
}
```

RiskScoreCard.vue maps `response.risk.final_score` to the dial.
RecommendationCard.vue maps `response.recommendations[]` to the coloured cards.

### Person 3 — Backend Developer

Replace the stub DB functions in `cloud_api.py`:

```python
# Replace the body of list_students() with:
rows = await db.fetch_all("SELECT * FROM students")
return {"students": [dict(r) for r in rows], "total": len(rows)}

# Replace the body of get_student() with:
row = await db.fetch_one("SELECT * FROM students WHERE id = :id",
                          values={"id": student_id})
if not row:
    raise HTTPException(status_code=404, detail="Student not found")
return dict(row)
```

### Person 5 — Integration & QA Lead

```bash
# Health check (use in CI/CD pipeline)
curl http://localhost:8000/health

# Run full test suite
python test_edusight.py all Maths.csv

# Trigger model retraining after new data upload
curl -X POST http://localhost:8000/ml/model/train \
  -H "Content-Type: application/json" \
  -d '{"csv_paths": ["Maths.csv"], "n_epochs": 40}'
```

---

## requirements.txt

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
openpyxl>=3.1.0
pennylane>=0.36.0
pennylane-lightning>=0.36.0
fastapi>=0.110.0
uvicorn>=0.29.0
httpx>=0.27.0
pydantic>=2.0.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `Model not found` on first run | Run `python ml_engine.py Maths.csv` first |
| `QRC weights not found` | Run `python quantum_engine.py Maths.csv` first |
| `pennylane not found` | `pip install pennylane pennylane-lightning` |
| CORS error from Vue | Add your frontend URL to `EDUSIGHT_CORS_ORIGINS` in `.env` |
| Slow first `/simulate` request | Engines are lazy-loaded on first call — warm up with `GET /health` |
| Cloud ML timeout | Increase `EDUSIGHT_TIMEOUT` in `.env` or check service URL |
| `Maths.csv` read error | File is Excel format disguised as .csv — `ml_engine.py` handles this automatically |

---

## Engine Summary

| Engine | Algorithm | Input | Output | File |
|--------|-----------|-------|--------|------|
| RiskEngine | Weighted formula + logistic | StudentProfile | Risk score 0–100 | `risk_engine.py` |
| MLEngine | Random Forest + Gradient Boosting + LR blend | CSV feature dict | ML risk score | `ml_engine.py` |
| QuantumRiskClassifier | Variational Quantum Circuit (4 qubits) | 4 features | Quantum risk score | `quantum_engine.py` |
| QuantumInterventionOptimizer | QAOA-inspired variational search | Budget + student | Optimal slider values | `quantum_engine.py` |
| RecommendationEngine | Threshold rules + sim re-ranking | RiskScore + SimResult | Intervention cards | `recommendation_engine.py` |
| CloudAPI | FastAPI REST + async httpx | HTTP JSON | HTTP JSON | `cloud_api.py` |
