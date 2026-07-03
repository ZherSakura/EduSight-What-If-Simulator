from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

_ML_ENGINE_DIR = Path(__file__).resolve().parent.parent / "Data_ML_Engine"
if _ML_ENGINE_DIR.is_dir() and str(_ML_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_ENGINE_DIR))

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from database import close_db, get_collection, init_db
from mapper import profile_to_api_response, row_to_student_profile
from models import (
    SimulateRequest,
    SimulateResponse,
    StudentDetail,
    StudentSummary,
    RecommendationItem,
    FactorWeights,
)

# ML / Risk / Recommendation engines
from risk_engine import RiskEngine, SimulationInput, Trend
from recommendation_engine import RecommendationEngine

ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "edusight_ml_model.joblib")
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://localhost:8001").rstrip("/")
CLOUD_API_TIMEOUT = float(os.getenv("CLOUD_API_TIMEOUT", "5.0"))


async def _fetch_ml_score(student_id: str, ml_data: dict | None) -> tuple[Optional[float], str]:
    if not ml_data:
        return None, "none"
    try:
        async with httpx.AsyncClient(timeout=CLOUD_API_TIMEOUT) as client:
            resp = await client.post(
                f"{CLOUD_API_URL}/ml/predict",
                json={"student_id": student_id, "student_data": ml_data},
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get("ml_risk_score")
                if raw is None:
                    return None, "none"
                return float(raw), data.get("source", "cloud")
            print(f"[ML] /ml/predict returned {resp.status_code} — using rule-based only")
            return None, "none"
    except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError) as e:
        print(f"[ML] Cloud service unreachable ({type(e).__name__}); using rule-based only")
        return None, "none"
    except Exception as e:
        print(f"[ML] Unexpected error calling /ml/predict: {e}")
        return None, "none"

# ---- App Lifecycle ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[EduSight] Starting up")
    await init_db()

    # Initialize engines
    app.state.risk_engine = RiskEngine(ml_model_path=ML_MODEL_PATH)
    app.state.rec_engine = RecommendationEngine()
    print("[EduSight] Engines ready")

    yield

    # Shutdown
    await close_db()
    print("[EduSight] Shutting down")

# -----  App  -----------------------------------------------------------
app = FastAPI(
    title="EduSight Backend API",
    description="Dropout risk prediction and intervention simulation for Malaysian schools.",
    version = "1.0.0",
    lifespan = lifespan,
)

# Allow oncoming requests from frontend dev server and production build.
# In production, restrict origins to the actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----  Helpers  -----------------------------------------------------------
def _get_engines(request):
    return request.app.state.risk_engine, request.app.state.rec_engine

def _build_sim_input(inputs) -> SimulationInput:
    # Converts frontend SimulationInputs to a RiskEngine SimulationInput
    # The frontend seeds absolute values, but the RiskEngine expects boost deltas.
    # The student's current values are stored in the profile, so the delta is
    # computed at call time inside the /simulate endpoint.

    # Attendance and grades deltas are computed in the endpoint where we have access
    # to the student's current values.
    return SimulationInput(
        attendance_boost = inputs.attendance,
        academic_boost = inputs.grades,
        counselling_sessions = inputs.counselling,
        welfare_support = inputs.welfare,
    )

def _format_weights(factor_weights: dict) -> FactorWeights:
    # Maps RiskEngine factor weights to the frontend FactorWeights.
    return FactorWeights(
        attendance = factor_weights.get("attendance", 0.0),
        academic = factor_weights.get("academic_performance", 0.0),
        socioeconomic = factor_weights.get("socioeconomic_status", 0.0),
        family = factor_weights.get("family_support", 0.0),
    )

def _format_score_change(delta: float) -> str:
    if delta < 0:
        return f"▼ {abs(delta):.0f} pts improvement from current score"
    if delta > 0:
        return f"▲ {delta:.0f} pts worse from current score"
    return "No change from current score"

def _level_label(risk_level: str) -> str:
    return {"low": "Low Risk", "medium": "Medium Risk", "high": "High Risk"}.get(risk_level, risk_level.title())

# -----  Routes  -----------------------------------------------------------
@app.get("/", tags=["health"])
async def root():
    # Health check
    return {"status": "ok", "service": "EduSight Backend API"}

@app.get(
    "/students",
    response_model = list[StudentSummary],
    tags = ["students"],
    summary = "List all students with current risk scores",
)
async def get_students(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200, description="Max students to return"),
    skip: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    # Returns all students with their current risk scores computed by RiskEngine.
    # Supports pagination via limit/skip query parameters.

    risk_engine, _ = _get_engines(request)
    collection = get_collection("students")

    cursor = collection.find({}, {"_id": 0}).skip(skip).limit(limit)
    docs = await cursor.to_list(length=limit)

    results = []
    for doc in docs:
        try:
            profile = row_to_student_profile(doc, doc["student_id"])
            risk_score = risk_engine.score(profile)
            results.append(profile_to_api_response(profile, risk_score, doc))
        except Exception as e:
            # Skip malformed rows rather than failing the whole list
            print(f"[/students] Skipping {doc.get('student_id')} due to {e}")

    return results

@app.get(
    "/students/{student_id}",
    response_model = StudentDetail,
    tags = ["students"],
    summary = "Get a single student profile with current risk score",
)
async def get_student(request: Request, student_id: str):
    # Returns one student's profile together with the latest risk assessment.
    risk_engine, _ = _get_engines(request)
    collection = get_collection("students")

    doc = await collection.find_one({"student_id": student_id}, {"_id": 0})
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    profile = row_to_student_profile(doc, student_id)
    risk_score = risk_engine.score(profile)

    return profile_to_api_response(profile, risk_score, doc)

@app.post(
    "/simulate",
    response_model = SimulateResponse,
    tags = ["simulation"],
    summary = "Run What-If simulation and return projected risk + recommended interventions",
)
async def simulate(request: Request, body: SimulateRequest):
    # Accepts the student ID and slider values from What-If Simulator.
    """
    Backend:
        1. Fetches the student's current profile from MongoDB
        2. Computes the risk score via RiskEngine
        3. Converts the slider values into boost deltas
        4. Runs RiskEngine.simulate() to get the projected score
        5. Calls RecommendationEngine.recommend() for recommended interventions
        6. Returns the full SimulateResponse to frontend
    """
    risk_engine, rec_engine = _get_engines(request)
    collection = get_collection("students")

    # -----  1. Fetch student  ------------------------------------------------------------
    doc = await collection.find_one({"student_id": body.studentId}, {"_id":0})
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Student '{body.studentId}' not found")

    profile = row_to_student_profile(doc, body.studentId)

    # -----  2. Risk score  ---------------------------------------------------------------
    baseline_risk = risk_engine.score(profile)

    # -----  3. Build simulation deltas  --------------------------------------------------
    # Frontend sends absolute values; RiskEngine expects deltas (boost amounts)
    attendance_boost = max(0.0, body.inputs.attendance - profile.attendance_rate)
    academic_boost = max(0.0, body.inputs.grades - profile.academic_score)
    welfare_normalized = body.inputs.welfare / 2.0

    sim_input = SimulationInput(
        attendance_boost = attendance_boost,
        academic_boost = academic_boost,
        counselling_sessions = body.inputs.counselling,
        welfare_support = welfare_normalized,
    )

    # -----  4. Simulate  ------------------------------------------------------------------
    sim_result = risk_engine.simulate(profile, sim_input)

    # -----  5. Recommendations  -----------------------------------------------------------
    report = rec_engine.recommend(
        student = profile,
        risk_score = baseline_risk,
        sim_input = sim_input,
        sim_result = sim_result,
    )

    # -----  6. Cloud ML score   -----------------------------------------------------------
    ml_score, ml_source = await _fetch_ml_score(body.studentId, profile.ml_data)

    # -----  7. Format response  -----------------------------------------------------------
    recommendations = [
        RecommendationItem(
            category = r.category.value,
            urgency = r.urgency.value,
            title = r.title,
            description = r.description,
            expected_impact = r.expected_impact,
        )
        for r in report.recommendations
    ]

    projected_risk_level = sim_result.risk_level_projected.value

    return SimulateResponse(
        baselineScore = sim_result.baseline_score,
        projectedScore = sim_result.projected_score,
        riskLevel = projected_risk_level,
        riskLabel = _level_label(projected_risk_level),
        scoreChangeText = _format_score_change(sim_result.score_delta),
        dropoutProbability = round(sim_result.projected_prob_3m * 100, 1),
        insightText = sim_result.summary,
        narrative = report.narrative,
        weights = _format_weights(sim_result.factor_weights),
        recommendations = recommendations,
        mlScore = ml_score,
        mlSource = ml_source,
    )