"""
EduSight — Cloud API Layer
============================
Manages all communication between the Vue.js frontend and the cloud-hosted ML/quantum services.  Built with FastAPI.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── load environment variables from .env if present ──────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
#  Environment configuration
# ─────────────────────────────────────────────

ENV                  = os.getenv("EDUSIGHT_ENV", "local")
ML_SERVICE_URL       = os.getenv("EDUSIGHT_ML_SERVICE_URL", "")
API_KEY              = os.getenv("EDUSIGHT_API_KEY", "")
ML_MODEL_PATH        = os.getenv("EDUSIGHT_ML_MODEL_PATH", "edusight_ml_model.joblib")
QRC_WEIGHTS_PATH     = os.getenv("EDUSIGHT_QRC_PATH", "edusight_qrc_weights.npy")
REQUEST_TIMEOUT_SECS = float(os.getenv("EDUSIGHT_TIMEOUT", "10"))
ALLOWED_ORIGINS      = os.getenv("EDUSIGHT_CORS_ORIGINS",
                                  "http://localhost:5173,http://localhost:3000").split(",")


# ─────────────────────────────────────────────
#  Pydantic request / response models
#  (these are what the Vue frontend sends/receives)
# ─────────────────────────────────────────────

class StudentProfileRequest(BaseModel):
    """Matches StudentProfile in risk_engine.py, serialisable over HTTP."""
    student_id     : str
    name           : str
    grade          : str
    attendance_rate: float = Field(..., ge=0, le=100)
    academic_score : float = Field(..., ge=0, le=100)
    socio_score    : float = Field(..., ge=0, le=100)
    family_support : float = Field(60.0, ge=0, le=100)
    trend          : str   = Field("stable",
                                    pattern="^(improving|stable|worsening)$")
    ml_data        : Optional[dict] = None   # raw CSV columns for ML/quantum


class SimulationRequest(BaseModel):
    """Body for POST /simulate — mirrors the 4 What-If sliders."""
    student         : StudentProfileRequest
    attendance_boost    : float = Field(0.0, ge=0, le=30)
    academic_boost      : float = Field(0.0, ge=0, le=25)
    counselling_sessions: int   = Field(0,   ge=0, le=10)
    welfare_support     : float = Field(0.0, ge=0, le=1)
    use_ml              : bool  = True    # include ML score in blend
    use_quantum         : bool  = False   # include quantum score in blend


class QuantumOptimiseRequest(BaseModel):
    """Body for POST /simulate/quantum."""
    student : StudentProfileRequest
    budget  : float = Field(40.0, ge=5, le=100)
    n_steps : int   = Field(30,   ge=5, le=100)


class MLPredictRequest(BaseModel):
    """Body for POST /ml/predict — raw CSV feature dict."""
    student_id  : str
    student_data: dict   # raw CSV columns (G1, G2, absences, failures, etc.)


class TrainRequest(BaseModel):
    """Body for POST /ml/model/train."""
    csv_paths  : list[str]   # paths on the server where CSVs are stored
    n_epochs   : int = 40    # quantum training epochs
    force_retrain: bool = False


# ─────────────────────────────────────────────
#  Cloud ML client
#  Handles outbound HTTP to the remote ML service
# ─────────────────────────────────────────────

class CloudMLClient:
    """
    Async HTTP client for the cloud-hosted ML service.

    When EDUSIGHT_ML_SERVICE_URL is set in .env, all ML inference
    requests are forwarded to the remote service.  If the service
    is unreachable or returns an error, the client automatically
    falls back to the local MLEngine.

    Cloud request format
    ────────────────────
    POST {ML_SERVICE_URL}/predict
    Headers: X-API-Key: {EDUSIGHT_API_KEY}
    Body:    { "student_id": "...", "student_data": {...} }

    Response: { "ml_risk_score": 75.2, "dropout_probability": 0.71, ... }
    """

    def __init__(self):
        self._base_url = ML_SERVICE_URL.rstrip("/")
        self._headers  = {
            "Content-Type": "application/json",
            "X-API-Key"   : API_KEY,
        }
        self._available: Optional[bool] = None   # None = not yet checked

    @property
    def is_configured(self) -> bool:
        return bool(self._base_url)

    async def health_check(self) -> bool:
        """Ping the cloud service. Returns True if reachable."""
        if not self.is_configured:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/health",
                    headers=self._headers,
                )
                self._available = resp.status_code == 200
                return self._available
        except Exception:
            self._available = False
            return False

    async def predict(self, student_id: str, student_data: dict) -> Optional[dict]:
        """
        Forward a prediction request to the cloud ML service.
        Returns the parsed response dict, or None if unavailable.
        """
        if not self.is_configured:
            return None
        try:
            payload = {"student_id": student_id, "student_data": student_data}
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECS) as client:
                resp = await client.post(
                    f"{self._base_url}/predict",
                    headers=self._headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    return resp.json()
                print(f"[CloudMLClient] Remote service returned {resp.status_code}")
                return None
        except httpx.TimeoutException:
            print(f"[CloudMLClient] Timeout after {REQUEST_TIMEOUT_SECS}s — "
                  f"falling back to local engine.")
            return None
        except Exception as e:
            print(f"[CloudMLClient] Request failed: {e} — falling back to local.")
            return None

    async def trigger_training(self, csv_paths: list[str], **kwargs) -> Optional[dict]:
        """
        Ask the cloud service to (re)train the ML model.
        Returns a job status dict, or None if unavailable.
        """
        if not self.is_configured:
            return None
        try:
            payload = {"csv_paths": csv_paths, **kwargs}
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self._base_url}/train",
                    headers=self._headers,
                    json=payload,
                )
                return resp.json() if resp.status_code == 200 else None
        except Exception as e:
            print(f"[CloudMLClient] Training request failed: {e}")
            return None


# ─────────────────────────────────────────────
#  Engine registry
#  Lazy-loads engines once on first request
# ─────────────────────────────────────────────

class EngineRegistry:
    """
    Singleton that holds loaded engine instances.
    Lazy-loads on first use so startup is fast.
    """

    def __init__(self):
        self._risk_engine   = None
        self._rec_engine    = None
        self._ml_engine     = None
        self._qrc           = None
        self._qio           = None
        self._cloud_client  = CloudMLClient()

    @property
    def cloud(self) -> CloudMLClient:
        return self._cloud_client

    def risk(self):
        if self._risk_engine is None:
            from risk_engine import RiskEngine
            ml_path = ML_MODEL_PATH if Path(ML_MODEL_PATH).exists() else None
            self._risk_engine = RiskEngine(ml_model_path=ml_path)
        return self._risk_engine

    def rec(self):
        if self._rec_engine is None:
            from recommendation_engine import RecommendationEngine
            self._rec_engine = RecommendationEngine()
        return self._rec_engine

    def ml(self):
        if self._ml_engine is None:
            if not Path(ML_MODEL_PATH).exists():
                return None
            try:
                from ml_engine import MLEngine
                self._ml_engine = MLEngine.load(ML_MODEL_PATH)
            except Exception as e:
                print(f"[EngineRegistry] Could not load ML model: {e}")
                return None
        return self._ml_engine

    def qrc(self):
        if self._qrc is None:
            if not Path(QRC_WEIGHTS_PATH).exists():
                return None
            try:
                from quantum_engine import QuantumRiskClassifier
                self._qrc = QuantumRiskClassifier.load(QRC_WEIGHTS_PATH)
            except Exception as e:
                print(f"[EngineRegistry] Could not load QRC: {e}")
                return None
        return self._qrc

    def qio(self):
        if self._qio is None:
            try:
                from quantum_engine import QuantumInterventionOptimizer
                self._qio = QuantumInterventionOptimizer()
            except Exception as e:
                print(f"[EngineRegistry] Could not init QIO: {e}")
                return None
        return self._qio

    def invalidate_ml(self):
        """Call after retraining to force reload on next request."""
        self._ml_engine  = None
        self._risk_engine = None   # also reload since it embeds ML


# Global registry (one instance per process)
registry = EngineRegistry()


# ─────────────────────────────────────────────
#  Helper — build internal StudentProfile
# ─────────────────────────────────────────────

def _build_student_profile(req: StudentProfileRequest):
    from risk_engine import StudentProfile, Trend
    return StudentProfile(
        student_id      = req.student_id,
        name            = req.student.name if hasattr(req, "student") else req.name,
        grade           = req.student.grade if hasattr(req, "student") else req.grade,
        attendance_rate = req.student.attendance_rate if hasattr(req, "student") else req.attendance_rate,
        academic_score  = req.student.academic_score  if hasattr(req, "student") else req.academic_score,
        socio_score     = req.student.socio_score     if hasattr(req, "student") else req.socio_score,
        family_support  = req.student.family_support  if hasattr(req, "student") else req.family_support,
        trend           = Trend(req.student.trend     if hasattr(req, "student") else req.trend),
        ml_data         = req.student.ml_data         if hasattr(req, "student") else req.ml_data,
    )

def _build_student(s: StudentProfileRequest):
    from risk_engine import StudentProfile, Trend
    return StudentProfile(
        student_id=s.student_id, name=s.name, grade=s.grade,
        attendance_rate=s.attendance_rate, academic_score=s.academic_score,
        socio_score=s.socio_score, family_support=s.family_support,
        trend=Trend(s.trend), ml_data=s.ml_data,
    )


# ─────────────────────────────────────────────
#  FastAPI application
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "EduSight API",
    description = "Cloud API layer for the EduSight dropout prevention system.",
    version     = "1.0.0",
    docs_url    = "/docs",      # Swagger UI at http://localhost:8000/docs
    redoc_url   = "/redoc",
)

# CORS — allow the Vue dev server (port 5173) and any configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS + ["*"] if ENV == "local" else ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
#  Middleware — request timing & logging
# ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start   = time.time()
    response= await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    print(f"[API] {request.method} {request.url.path}  "
          f"→ {response.status_code}  ({elapsed}ms)")
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ─────────────────────────────────────────────
#  Endpoint 1 — Health check
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """
    Service health check.
    Called by Person 5's CI/CD pipeline and the Vue frontend on startup.
    """
    cloud_ok = await registry.cloud.health_check()
    return {
        "status"         : "ok",
        "env"            : ENV,
        "ml_model_loaded": registry.ml() is not None,
        "qrc_loaded"     : registry.qrc() is not None,
        "cloud_ml_url"   : ML_SERVICE_URL or "not configured",
        "cloud_reachable": cloud_ok,
        "timestamp"      : time.time(),
    }


# ─────────────────────────────────────────────
#  Endpoint 2 — Student profile (stub)
#  Person 3 replaces these with real DB queries
# ─────────────────────────────────────────────

@app.get("/students", tags=["Students"])
async def list_students():
    """
    Return all student profiles.
    STUB — Person 3 replaces this body with a database query.
    """
    # TODO (Person 3): replace with  db.query("SELECT * FROM students")
    return {
        "students": [
            {
                "student_id": "STU-001", "name": "Muhammad Ali bin Faisal",
                "grade": "Form 4", "attendance_rate": 62.0,
                "academic_score": 49.0, "socio_score": 55.0,
                "family_support": 40.0, "trend": "worsening",
            },
            {
                "student_id": "STU-002", "name": "Siti Aisyah binti Rahman",
                "grade": "Form 3", "attendance_rate": 92.0,
                "academic_score": 78.0, "socio_score": 20.0,
                "family_support": 80.0, "trend": "improving",
            },
        ],
        "total": 2,
    }


@app.get("/students/{student_id}", tags=["Students"])
async def get_student(student_id: str):
    """
    Return a single student profile by ID.
    STUB — Person 3 replaces with  db.query("SELECT * FROM students WHERE id=?")
    """
    # TODO (Person 3): replace with DB lookup
    if student_id != "STU-001":
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")
    return {
        "student_id": "STU-001", "name": "Muhammad Ali bin Faisal",
        "grade": "Form 4", "attendance_rate": 62.0,
        "academic_score": 49.0, "socio_score": 55.0,
        "family_support": 40.0, "trend": "worsening",
        "ml_data": {
            "absences": 18, "failures": 2, "G1": 5, "G2": 4,
            "Dalc": 3, "Walc": 4, "school": "GP", "sex": "M",
            "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 2, "Fedu": 1, "studytime": 1, "higher": "no",
            "internet": "no", "romantic": "yes", "famrel": 2,
            "freetime": 4, "goout": 4, "health": 2,
        },
    }


# ─────────────────────────────────────────────
#  Endpoint 3 — What-If Simulation  ★ CORE ★
# ─────────────────────────────────────────────

@app.post("/simulate", tags=["Simulation"])
async def simulate(body: SimulationRequest):
    """
    Core What-If Simulator endpoint.

    Flow
    ────
    1. Build StudentProfile from request
    2. Run rule-based simulation via RiskEngine
    3. If use_ml=True: try cloud ML service first, fall back to local MLEngine
    4. If use_quantum=True: blend in quantum risk score
    5. Run RecommendationEngine with simulation context
    6. Return combined JSON response to Vue frontend

    Called by
    ─────────
    Person 2's AttendanceSlider.vue / GradeSlider.vue (on slider change)
    Person 1's StudentProfile.vue (on student load)
    """
    from risk_engine import SimulationInput

    student   = _build_student(body.student)
    sim_input = SimulationInput(
        attendance_boost     = body.attendance_boost,
        academic_boost       = body.academic_boost,
        counselling_sessions = body.counselling_sessions,
        welfare_support      = body.welfare_support,
    )

    risk_engine = registry.risk()
    rec_engine  = registry.rec()

    # ── Step 1: rule-based score + simulation ────────────────────
    risk_score = risk_engine.score(student)
    sim_result = risk_engine.simulate(student, sim_input)

    # ── Step 2: ML score (cloud-first, local fallback) ───────────
    ml_score      = None
    ml_source     = "none"
    cloud_ml_data = None

    if body.use_ml and body.student.ml_data:
        # Try cloud ML service first
        cloud_ml_data = await registry.cloud.predict(
            body.student.student_id, body.student.ml_data
        )
        if cloud_ml_data:
            ml_score  = cloud_ml_data.get("ml_risk_score")
            ml_source = "cloud"
        else:
            # Fall back to local MLEngine
            local_ml = registry.ml()
            if local_ml:
                try:
                    ml_pred  = local_ml.predict_risk(
                        body.student.student_id, body.student.ml_data
                    )
                    ml_score = ml_pred.ml_risk_score
                    ml_source = "local"
                except Exception as e:
                    print(f"[API] Local ML inference failed: {e}")

    # ── Step 3: quantum score (optional) ─────────────────────────
    quantum_score  = None
    quantum_source = "none"

    if body.use_quantum and body.student.ml_data:
        qrc = registry.qrc()
        if qrc:
            try:
                from quantum_engine import get_quantum_risk_score
                blended_q = get_quantum_risk_score(
                    qrc, body.student.ml_data, body.student.student_id,
                    rule_based_score=risk_score.total_score,
                    ml_score=ml_score,
                )
                quantum_score  = blended_q
                quantum_source = "local-quantum"
            except Exception as e:
                print(f"[API] Quantum inference failed: {e}")

    # ── Step 4: final blended score ───────────────────────────────
    final_score = risk_score.total_score
    if ml_score is not None and quantum_score is not None:
        final_score = 0.50 * risk_score.total_score + 0.30 * ml_score + 0.20 * quantum_score
    elif ml_score is not None:
        final_score = 0.55 * risk_score.total_score + 0.45 * ml_score
    elif quantum_score is not None:
        final_score = 0.80 * risk_score.total_score + 0.20 * quantum_score

    # ── Step 5: recommendations ───────────────────────────────────
    report = rec_engine.recommend(
        student, risk_score,
        sim_input=sim_input, sim_result=sim_result,
    )

    # ── Step 6: build response ────────────────────────────────────
    return {
        "student_id"        : body.student.student_id,
        "risk": {
            "rule_based_score"  : risk_score.total_score,
            "ml_score"          : ml_score,
            "quantum_score"     : quantum_score,
            "final_score"       : round(final_score, 1),
            "risk_level"        : risk_score.risk_level.value,
            "factor_weights"    : risk_score.factor_weights,
            "dropout_prob_3m"   : risk_score.dropout_prob_3m,
            "dropout_prob_6m"   : risk_score.dropout_prob_6m,
            "ml_source"         : ml_source,
            "quantum_source"    : quantum_source,
            "explanation"       : risk_score.explanation,
        },
        "simulation": {
            "baseline_score"    : sim_result.baseline_score,
            "projected_score"   : sim_result.projected_score,
            "score_delta"       : sim_result.score_delta,
            "baseline_prob_3m"  : sim_result.baseline_prob_3m,
            "projected_prob_3m" : sim_result.projected_prob_3m,
            "risk_level_baseline"  : sim_result.risk_level_baseline.value,
            "risk_level_projected" : sim_result.risk_level_projected.value,
            "dominant_factor"   : sim_result.dominant_factor,
            "summary"           : sim_result.summary,
        },
        "recommendations": [
            {
                "category"        : r.category.value,
                "urgency"         : r.urgency.value,
                "title"           : r.title,
                "description"     : r.description,
                "expected_impact" : r.expected_impact,
                "owner"           : r.owner,
                "icon"            : r.icon,
            }
            for r in report.recommendations
        ],
        "narrative"     : report.narrative,
        "priority_action": report.priority_action,
    }


# ─────────────────────────────────────────────
#  Endpoint 4 — Quantum Optimised Simulation
# ─────────────────────────────────────────────

@app.post("/simulate/quantum", tags=["Simulation"])
async def simulate_quantum(body: QuantumOptimiseRequest):
    """
    Quantum Intervention Optimizer endpoint.

    Runs the QuantumInterventionOptimizer to find the best slider
    combination within the given budget, then returns both the optimal
    plan AND a full simulation result for that plan.

    Called by the 'Quantum Optimise' button in the What-If Simulator UI.
    """
    student    = _build_student(body.student)
    risk_engine= registry.risk()
    rec_engine = registry.rec()
    qio        = registry.qio()

    if qio is None:
        raise HTTPException(
            status_code=503,
            detail="Quantum engine not available. "
                   "Ensure pennylane is installed: pip install pennylane"
        )

    # Run optimisation in a thread pool (CPU-bound)
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: qio.optimise(
            student, risk_engine,
            budget=body.budget, n_steps=body.n_steps, verbose=False,
        )
    )

    # Get full simulation + recommendations for the optimal plan
    from risk_engine import SimulationInput
    opt_input  = SimulationInput(
        attendance_boost     = result.attendance_boost,
        academic_boost       = result.academic_boost,
        counselling_sessions = result.counselling_sessions,
        welfare_support      = result.welfare_support,
    )
    risk_score = risk_engine.score(student)
    sim_result = risk_engine.simulate(student, opt_input)
    report     = rec_engine.recommend(
        student, risk_score,
        sim_input=opt_input, sim_result=sim_result,
    )

    return {
        "student_id"        : body.student.student_id,
        "quantum_optimal_plan": {
            "attendance_boost"    : result.attendance_boost,
            "academic_boost"      : result.academic_boost,
            "counselling_sessions": result.counselling_sessions,
            "welfare_support"     : result.welfare_support,
            "projected_risk_score": result.projected_risk_score,
            "risk_reduction"      : result.risk_reduction,
            "optimisation_steps"  : result.optimisation_steps,
            "method"              : result.method,
        },
        "simulation": {
            "baseline_score"    : sim_result.baseline_score,
            "projected_score"   : sim_result.projected_score,
            "score_delta"       : sim_result.score_delta,
            "projected_prob_3m" : sim_result.projected_prob_3m,
            "summary"           : sim_result.summary,
        },
        "recommendations": [
            {
                "urgency"    : r.urgency.value,
                "title"      : r.title,
                "description": r.description,
                "owner"      : r.owner,
            }
            for r in report.recommendations
        ],
    }


# ─────────────────────────────────────────────
#  Endpoint 5 — Raw ML prediction
# ─────────────────────────────────────────────

@app.post("/ml/predict", tags=["ML Service"])
async def ml_predict(body: MLPredictRequest):
    """
    Raw ML risk prediction for a single student.

    Cloud-first: forwards to the remote ML service if configured.
    Falls back to local MLEngine if cloud is unavailable.
    """
    # Try cloud first
    cloud_result = await registry.cloud.predict(body.student_id, body.student_data)
    if cloud_result:
        return {**cloud_result, "source": "cloud"}

    # Local fallback
    local_ml = registry.ml()
    if local_ml is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Train first via POST /ml/model/train"
        )

    try:
        pred = local_ml.predict_risk(body.student_id, body.student_data)
        return {
            "student_id"          : pred.student_id,
            "dropout_probability" : pred.dropout_probability,
            "atrisk_probability"  : pred.atrisk_probability,
            "predicted_g3"        : pred.predicted_g3,
            "ml_risk_score"       : pred.ml_risk_score,
            "top_risk_features"   : pred.top_risk_features,
            "confidence"          : pred.confidence,
            "source"              : "local",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML inference error: {e}")


# ─────────────────────────────────────────────
#  Endpoint 6 — ML model status
# ─────────────────────────────────────────────

@app.get("/ml/model/status", tags=["ML Service"])
async def ml_model_status():
    """
    Check if a trained ML model is available (local and/or cloud).
    Called by Person 5's deployment health checks.
    """
    local_exists = Path(ML_MODEL_PATH).exists()
    cloud_ok     = await registry.cloud.health_check()

    return {
        "local_model_exists" : local_exists,
        "local_model_path"   : ML_MODEL_PATH,
        "cloud_configured"   : registry.cloud.is_configured,
        "cloud_reachable"    : cloud_ok,
        "qrc_weights_exist"  : Path(QRC_WEIGHTS_PATH).exists(),
        "active_source"      : (
            "cloud" if cloud_ok else
            "local" if local_exists else
            "none — train the model first"
        ),
    }


# ─────────────────────────────────────────────
#  Endpoint 7 — Trigger model training
# ─────────────────────────────────────────────

@app.post("/ml/model/train", tags=["ML Service"])
async def ml_model_train(body: TrainRequest):
    """
    Trigger ML model (re)training.

    If the cloud service is configured, the training job is dispatched
    to the cloud (async — returns a job ID immediately).
    Otherwise, trains locally in a background thread.

    Person 5 calls this endpoint from the CI/CD pipeline whenever
    new student data is added to the database.
    """
    # Try to delegate to cloud training service
    cloud_job = await registry.cloud.trigger_training(
        body.csv_paths,
        n_epochs=body.n_epochs,
        force_retrain=body.force_retrain,
    )
    if cloud_job:
        return {
            "status" : "dispatched",
            "source" : "cloud",
            "job"    : cloud_job,
            "message": "Training job dispatched to cloud ML service.",
        }

    # Local training (run in thread pool to avoid blocking the API)
    missing = [p for p in body.csv_paths if not Path(p).exists()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV files not found on server: {missing}"
        )

    loop = asyncio.get_event_loop()

    async def _train():
        from ml_engine import MLEngine
        engine = MLEngine()
        await loop.run_in_executor(None, lambda: engine.train(*body.csv_paths))
        await loop.run_in_executor(None, lambda: engine.save(ML_MODEL_PATH))
        registry.invalidate_ml()   # force reload on next request
        return {"status": "complete", "source": "local", "path": ML_MODEL_PATH}

    # Fire and forget — return immediately, training runs in background
    asyncio.create_task(_train())
    return {
        "status" : "started",
        "source" : "local",
        "message": f"Local training started on {body.csv_paths}. "
                   f"Model will be saved to {ML_MODEL_PATH}.",
    }


# ─────────────────────────────────────────────
#  Endpoint 8 — Recommendations
# ─────────────────────────────────────────────

@app.get("/recommendations/{student_id}", tags=["Recommendations"])
async def get_recommendations(student_id: str):
    """
    Get baseline recommendations for a student (no simulation context).
    Called by Person 2's RecommendationCard.vue on initial page load.

    For simulation-context recommendations, use POST /simulate instead.
    """
    # Fetch student profile (Person 3 will replace with DB call)
    try:
        student_data = await get_student(student_id)
    except HTTPException:
        raise HTTPException(status_code=404,
                            detail=f"Student {student_id} not found.")

    from risk_engine import StudentProfile, Trend
    student = StudentProfile(
        student_id      = student_data["student_id"],
        name            = student_data["name"],
        grade           = student_data["grade"],
        attendance_rate = student_data["attendance_rate"],
        academic_score  = student_data["academic_score"],
        socio_score     = student_data["socio_score"],
        family_support  = student_data["family_support"],
        trend           = Trend(student_data["trend"]),
    )

    risk_score = registry.risk().score(student)
    report     = registry.rec().recommend(student, risk_score)

    return {
        "student_id"    : student_id,
        "risk_level"    : report.risk_level.value,
        "narrative"     : report.narrative,
        "priority_action": report.priority_action,
        "recommendations": [
            {
                "category"       : r.category.value,
                "urgency"        : r.urgency.value,
                "title"          : r.title,
                "description"    : r.description,
                "expected_impact": r.expected_impact,
                "owner"          : r.owner,
                "icon"           : r.icon,
            }
            for r in report.recommendations
        ],
    }


# ─────────────────────────────────────────────
#  Startup event
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 55)
    print("  EduSight API starting up...")
    print(f"  Environment     : {ENV}")
    print(f"  ML model path   : {ML_MODEL_PATH}")
    print(f"  QRC weights     : {QRC_WEIGHTS_PATH}")
    print(f"  Cloud ML URL    : {ML_SERVICE_URL or 'not configured'}")
    print(f"  CORS origins    : {ALLOWED_ORIGINS}")
    print(f"  Swagger UI      : http://localhost:8000/docs")
    # Pre-load engines at startup to avoid first-request latency
    registry.risk()
    registry.rec()
    ml_loaded = registry.ml() is not None
    print(f"  ML model loaded : {ml_loaded}")
    cloud_ok  = await registry.cloud.health_check()
    print(f"  Cloud ML online : {cloud_ok}")
    print("=" * 55 + "\n")
