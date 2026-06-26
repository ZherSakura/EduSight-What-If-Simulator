"""
EduSight — Integration Test Suite
====================================
Tests all three engines together and individually.
"""

import sys
from pathlib import Path


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

_results = []

def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{PASS} {label}")
        _results.append(True)
    else:
        print(f"{FAIL} {label}  ← {detail}")
        _results.append(False)

def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────
#  Shared test student
# ─────────────────────────────────────────────

def make_high_risk_student():
    from risk_engine import StudentProfile, Trend
    return StudentProfile(
        student_id      = "STU-001",
        name            = "Muhammad Ali bin Faisal",
        grade           = "Form 4",
        attendance_rate = 62.0,
        academic_score  = 49.0,
        socio_score     = 55.0,
        family_support  = 40.0,
        trend           = Trend.WORSENING,
    )

def make_low_risk_student():
    from risk_engine import StudentProfile, Trend
    return StudentProfile(
        student_id      = "STU-002",
        name            = "Siti Aisyah binti Rahman",
        grade           = "Form 3",
        attendance_rate = 92.0,
        academic_score  = 78.0,
        socio_score     = 20.0,
        family_support  = 80.0,
        trend           = Trend.IMPROVING,
    )


# ─────────────────────────────────────────────
#  TEST SUITE A — Risk Engine (rule-based)
# ─────────────────────────────────────────────

def test_risk_engine_rule_based():
    section("TEST A — risk_engine.py  (rule-based, no ML)")
    from risk_engine import RiskEngine, RiskLevel, SimulationInput

    engine = RiskEngine(ml_model_path=None)

    # ── Score: high-risk student ──
    student = make_high_risk_student()
    score   = engine.score(student)

    check("risk_engine — score returns a value",
          0 < score.total_score < 100,
          f"got {score.total_score}")

    check("risk_engine — high-risk student classified HIGH",
          score.risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM),
          f"got {score.risk_level.value} — expected HIGH or MEDIUM")

    check("risk_engine — dropout_prob_3m is a probability",
          0 < score.dropout_prob_3m < 1,
          f"got {score.dropout_prob_3m}")

    check("risk_engine — factor_weights sum ~100",
          abs(sum(score.factor_weights.values()) - 100.0) < 1.0,
          f"sum = {sum(score.factor_weights.values()):.1f}")

    check("risk_engine — ml_enhanced is False (no model loaded)",
          score.ml_enhanced is False)

    # ── Score: low-risk student ──
    low_student  = make_low_risk_student()
    low_score    = engine.score(low_student)

    check("risk_engine — low-risk student scores lower than high-risk",
          low_score.total_score < score.total_score,
          f"low={low_score.total_score} vs high={score.total_score}")

    check("risk_engine — low-risk classified LOW or MEDIUM",
          low_score.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM),
          f"got {low_score.risk_level.value}")

    # ── Simulation ──
    sim_input = SimulationInput(
        attendance_boost=18.0, academic_boost=10.0,
        counselling_sessions=5, welfare_support=1.0,
    )
    result = engine.simulate(student, sim_input)

    check("risk_engine — simulation reduces risk score",
          result.score_delta < 0,
          f"delta = {result.score_delta}")

    check("risk_engine — projected score < baseline",
          result.projected_score < result.baseline_score,
          f"{result.projected_score} vs {result.baseline_score}")

    check("risk_engine — projected 3m prob < baseline",
          result.projected_prob_3m < result.baseline_prob_3m,
          f"{result.projected_prob_3m} vs {result.baseline_prob_3m}")

    check("risk_engine — summary string populated",
          len(result.summary) > 10)

    # ── No intervention → no change ──
    no_sim  = SimulationInput()
    no_result = engine.simulate(student, no_sim)
    check("risk_engine — zero intervention → score unchanged",
          abs(no_result.score_delta) < 0.5,
          f"delta = {no_result.score_delta}")

    print(f"\n{INFO} High-risk score : {score.total_score} ({score.risk_level.value})")
    print(f"{INFO} Low-risk  score : {low_score.total_score} ({low_score.risk_level.value})")
    print(f"{INFO} Sim baseline    : {result.baseline_score}  →  projected: {result.projected_score}")


# ─────────────────────────────────────────────
#  TEST SUITE B — Recommendation Engine
# ─────────────────────────────────────────────

def test_recommendation_engine():
    section("TEST B — recommendation_engine.py")
    from risk_engine import RiskEngine, RiskLevel, SimulationInput
    from recommendation_engine import RecommendationEngine, UrgencyLevel

    risk_engine = RiskEngine(ml_model_path=None)
    rec_engine  = RecommendationEngine()
    student     = make_high_risk_student()
    risk_score  = risk_engine.score(student)
    sim_input   = SimulationInput(attendance_boost=18.0, academic_boost=10.0,
                                   counselling_sessions=5, welfare_support=1.0)
    sim_result  = risk_engine.simulate(student, sim_input)

    # ── Without simulation ──
    report_base = rec_engine.recommend(student, risk_score)

    check("rec_engine — returns recommendations",
          len(report_base.recommendations) > 0,
          f"got {len(report_base.recommendations)}")

    check("rec_engine — priority_action is populated",
          len(report_base.priority_action) > 5)

    check("rec_engine — narrative mentions student name",
          student.name in report_base.narrative)

    check("rec_engine — no duplicate categories",
          len(report_base.recommendations) ==
          len({r.category for r in report_base.recommendations}))

    check("rec_engine — high-risk gets CRITICAL or HIGH urgency first",
          report_base.recommendations[0].urgency in
          (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH))

    # ── With simulation ──
    report_sim = rec_engine.recommend(student, risk_score,
                                       sim_input=sim_input, sim_result=sim_result)

    check("rec_engine — simulation-aware report has recommendations",
          len(report_sim.recommendations) > 0)

    check("rec_engine — narrative updated with sim reduction",
          "%" in report_sim.narrative)

    # ── Low-risk student ──
    low_student = make_low_risk_student()
    low_score   = risk_engine.score(low_student)
    low_report  = rec_engine.recommend(low_student, low_score)

    check("rec_engine — low-risk student still gets monitoring rec",
          any(r.urgency.value == "routine" for r in low_report.recommendations))

    print(f"\n{INFO} Recommendations for high-risk student:")
    for r in report_sim.recommendations:
        print(f"   [{r.urgency.value.upper():<8}] {r.title}")


# ─────────────────────────────────────────────
#  TEST SUITE C — ML Engine
# ─────────────────────────────────────────────

def test_ml_engine(*csv_paths):
    section("TEST C — ml_engine.py  (requires CSV files)")
    from ml_engine import MLEngine, ml_risk_to_student_profile_override

    if not csv_paths:
        print(f"{INFO} No CSV path provided — skipping ML tests.")
        print(f"{INFO} To run: python test_edusight.py ml Maths.csv")
        return

    for p in csv_paths:
        check(f"ml_engine — CSV exists: {p}", Path(p).exists(),
              f"File not found: {p}")

    if not all(Path(p).exists() for p in csv_paths):
        print(f"{INFO} Skipping ML training — missing CSV files.")
        return

    # Train
    engine = MLEngine()
    engine.train(*csv_paths)
    check("ml_engine — training completed", engine._is_trained)

    # Evaluate
    metrics = engine.evaluate()
    check("ml_engine — accuracy > 0.70", metrics.accuracy > 0.70,
          f"accuracy = {metrics.accuracy}")
    check("ml_engine — ROC-AUC > 0.70", metrics.roc_auc > 0.70,
          f"roc_auc = {metrics.roc_auc}")
    check("ml_engine — feature importances populated",
          len(metrics.feature_importances) > 5)

    # Single inference
    student_data = {
        "school": "GP", "sex": "M", "age": 17,
        "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 2, "Fedu": 1, "Mjob": "other", "Fjob": "other",
        "reason": "home", "guardian": "mother",
        "traveltime": 2, "studytime": 1, "failures": 2,
        "schoolsup": "no", "famsup": "no", "paid": "no",
        "activities": "no", "nursery": "yes", "higher": "no",
        "internet": "no", "romantic": "yes",
        "famrel": 2, "freetime": 4, "goout": 4,
        "Dalc": 3, "Walc": 4, "health": 2, "absences": 18,
        "G1": 5, "G2": 4,
    }
    pred = engine.predict_risk("STU-001", student_data)
    check("ml_engine — dropout_probability in [0,1]",
          0 <= pred.dropout_probability <= 1,
          f"got {pred.dropout_probability}")
    check("ml_engine — ml_risk_score in [0,100]",
          0 <= pred.ml_risk_score <= 100,
          f"got {pred.ml_risk_score}")
    check("ml_engine — top_risk_features populated",
          len(pred.top_risk_features) > 0)
    check("ml_engine — confidence is 'high' (G1 & G2 present)",
          pred.confidence == "high")

    # Blend function
    blended = ml_risk_to_student_profile_override(pred, rule_based_score=82.0)
    check("ml_engine — blended score in [1,99]",
          1 <= blended <= 99, f"got {blended}")

    # Save & load
    engine.save("edusight_ml_model.joblib")
    engine2 = MLEngine.load("edusight_ml_model.joblib")
    pred2   = engine2.predict_risk("STU-001", student_data)
    check("ml_engine — save/load produces same score",
          abs(pred.ml_risk_score - pred2.ml_risk_score) < 0.1,
          f"{pred.ml_risk_score} vs {pred2.ml_risk_score}")

    print(f"\n{INFO} ML dropout probability : {pred.dropout_probability*100:.1f}%")
    print(f"{INFO} ML risk score           : {pred.ml_risk_score}")
    print(f"{INFO} Blended score (82 + ML) : {blended}")


# ─────────────────────────────────────────────
#  TEST SUITE D — Full pipeline (rule-based)
# ─────────────────────────────────────────────

def test_full_pipeline():
    section("TEST D — Full pipeline (all 3 engines together, rule-based)")
    from risk_engine import RiskEngine, SimulationInput, Trend
    from recommendation_engine import RecommendationEngine

    risk_engine = RiskEngine(ml_model_path=None)
    rec_engine  = RecommendationEngine()

    students = [make_high_risk_student(), make_low_risk_student()]

    for student in students:
        risk_score = risk_engine.score(student)
        sim_input  = SimulationInput(attendance_boost=15.0, academic_boost=8.0,
                                      counselling_sessions=3, welfare_support=0.5)
        sim_result = risk_engine.simulate(student, sim_input)
        report     = rec_engine.recommend(student, risk_score,
                                           sim_input=sim_input, sim_result=sim_result)

        check(f"pipeline — {student.name[:20]} — score computed",
              0 < risk_score.total_score < 100)
        check(f"pipeline — {student.name[:20]} — simulation ran",
              sim_result.baseline_score > 0)
        check(f"pipeline — {student.name[:20]} — recommendations generated",
              len(report.recommendations) > 0)
        check(f"pipeline — {student.name[:20]} — narrative generated",
              len(report.narrative) > 20)

    print(f"\n{INFO} Full pipeline complete for {len(students)} students.")


# ─────────────────────────────────────────────
#  TEST SUITE E — ML-enhanced pipeline
# ─────────────────────────────────────────────

def test_ml_enhanced_pipeline():
    """Only runs if edusight_ml_model.joblib already exists on disk."""
    section("TEST E — ML-enhanced pipeline (needs saved model)")
    model_path = "edusight_ml_model.joblib"

    if not Path(model_path).exists():
        print(f"{INFO} '{model_path}' not found — skipping.")
        print(f"{INFO} Run TEST C first:  python test_edusight.py ml Maths.csv")
        return

    from risk_engine import RiskEngine, SimulationInput, Trend, StudentProfile
    from recommendation_engine import RecommendationEngine

    # Student WITH raw ML feature data
    student_with_ml = StudentProfile(
        student_id      = "STU-001",
        name            = "Muhammad Ali bin Faisal",
        grade           = "Form 4",
        attendance_rate = 62.0,
        academic_score  = 49.0,
        socio_score     = 55.0,
        family_support  = 40.0,
        trend           = Trend.WORSENING,
        ml_data         = {   # ← raw CSV columns for ML model
            "school": "GP", "sex": "M", "age": 17,
            "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 2, "Fedu": 1, "Mjob": "other", "Fjob": "other",
            "reason": "home", "guardian": "mother",
            "traveltime": 2, "studytime": 1, "failures": 2,
            "schoolsup": "no", "famsup": "no", "paid": "no",
            "activities": "no", "nursery": "yes", "higher": "no",
            "internet": "no", "romantic": "yes",
            "famrel": 2, "freetime": 4, "goout": 4,
            "Dalc": 3, "Walc": 4, "health": 2, "absences": 18,
            "G1": 5, "G2": 4,
        }
    )

    risk_engine = RiskEngine(ml_model_path=model_path)
    rec_engine  = RecommendationEngine()

    score_ml = risk_engine.score(student_with_ml)
    check("ML-enhanced pipeline — ml_enhanced flag is True",
          score_ml.ml_enhanced is True,
          "ML model loaded but ml_enhanced still False")
    check("ML-enhanced pipeline — score in valid range",
          0 < score_ml.total_score < 100,
          f"got {score_ml.total_score}")

    sim_result = risk_engine.simulate(student_with_ml,
                    SimulationInput(attendance_boost=18.0, academic_boost=10.0,
                                    counselling_sessions=5, welfare_support=1.0))
    report = rec_engine.recommend(student_with_ml, score_ml,
                                   sim_result=sim_result)

    check("ML-enhanced pipeline — recommendations generated",
          len(report.recommendations) > 0)

    print(f"\n{INFO} ML-enhanced score  : {score_ml.total_score}")
    print(f"{INFO} ML enhanced flag   : {score_ml.ml_enhanced}")
    print(f"{INFO} Explanation        : {score_ml.explanation}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
#  TEST SUITE G — Cloud API Layer
# ─────────────────────────────────────────────

def test_cloud_api():
    section("TEST G — cloud_api.py  (FastAPI endpoints, no server needed)")
    import asyncio
    from cloud_api import (
        app, registry, CloudMLClient, EngineRegistry,
        SimulationRequest, StudentProfileRequest,
        QuantumOptimiseRequest, MLPredictRequest, TrainRequest,
        simulate, health, ml_model_status, get_student,
        get_recommendations, ml_predict,
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # ── Health check ──────────────────────────────────────────────
    resp = client.get("/health")
    check("cloud_api — /health returns 200",
          resp.status_code == 200, f"got {resp.status_code}")
    data = resp.json()
    check("cloud_api — /health has status=ok",
          data.get("status") == "ok", str(data))
    check("cloud_api — /health has env field",
          "env" in data)
    check("cloud_api — /health has ml_model_loaded field",
          "ml_model_loaded" in data)

    # ── Student endpoints ─────────────────────────────────────────
    resp = client.get("/students")
    check("cloud_api — GET /students returns 200",
          resp.status_code == 200)
    students = resp.json()
    check("cloud_api — GET /students returns list",
          "students" in students and len(students["students"]) > 0)

    resp = client.get("/students/STU-001")
    check("cloud_api — GET /students/STU-001 returns 200",
          resp.status_code == 200)
    stu = resp.json()
    check("cloud_api — student has required fields",
          all(k in stu for k in ["student_id","name","grade",
                                   "attendance_rate","academic_score"]))

    resp = client.get("/students/DOES-NOT-EXIST")
    check("cloud_api — unknown student returns 404",
          resp.status_code == 404)

    # ── POST /simulate (core endpoint) ───────────────────────────
    sim_payload = {
        "student": {
            "student_id"     : "STU-001",
            "name"           : "Muhammad Ali bin Faisal",
            "grade"          : "Form 4",
            "attendance_rate" : 62.0,
            "academic_score"  : 49.0,
            "socio_score"     : 55.0,
            "family_support"  : 40.0,
            "trend"           : "worsening",
            "ml_data"         : None,
        },
        "attendance_boost"    : 18.0,
        "academic_boost"      : 10.0,
        "counselling_sessions" : 5,
        "welfare_support"      : 1.0,
        "use_ml"              : False,
        "use_quantum"         : False,
    }
    resp = client.post("/simulate", json=sim_payload)
    check("cloud_api — POST /simulate returns 200",
          resp.status_code == 200, resp.text[:200])
    sim = resp.json()
    check("cloud_api — /simulate has risk block",
          "risk" in sim)
    check("cloud_api — /simulate has simulation block",
          "simulation" in sim)
    check("cloud_api — /simulate has recommendations",
          "recommendations" in sim and len(sim["recommendations"]) > 0)
    check("cloud_api — /simulate projected < baseline",
          sim["simulation"]["projected_score"] < sim["simulation"]["baseline_score"],
          f"proj={sim["simulation"]["projected_score"]} base={sim["simulation"]["baseline_score"]}")
    check("cloud_api — /simulate risk_level field present",
          "risk_level" in sim["risk"])
    check("cloud_api — /simulate narrative present",
          len(sim.get("narrative", "")) > 10)

    # Validate slider=0 → no change
    zero_payload = {**sim_payload,
                    "attendance_boost": 0, "academic_boost": 0,
                    "counselling_sessions": 0, "welfare_support": 0}
    resp0 = client.post("/simulate", json=zero_payload)
    check("cloud_api — /simulate zero sliders → delta near 0",
          resp0.status_code == 200 and
          abs(resp0.json()["simulation"]["score_delta"]) < 1.0)

    # ── POST /ml/predict ─────────────────────────────────────────
    ml_payload = {
        "student_id": "STU-001",
        "student_data": {
            "absences": 18, "failures": 2, "G1": 5, "G2": 4,
            "Dalc": 3, "Walc": 4, "school": "GP", "sex": "M",
            "age": 17, "address": "U", "famsize": "GT3",
            "Pstatus": "T", "Medu": 2, "Fedu": 1,
            "studytime": 1, "higher": "no", "internet": "no",
            "romantic": "yes", "famrel": 2, "freetime": 4,
            "goout": 4, "health": 2, "Mjob": "other",
            "Fjob": "other", "reason": "home", "guardian": "mother",
            "traveltime": 2, "schoolsup": "no", "famsup": "no",
            "paid": "no", "activities": "no", "nursery": "yes",
            "famrel": 2
        },
    }
    resp = client.post("/ml/predict", json=ml_payload)
    # 200 if model loaded, 503 if not — both are valid
    check("cloud_api — POST /ml/predict returns 200 or 503",
          resp.status_code in (200, 503),
          f"got {resp.status_code}")
    if resp.status_code == 200:
        ml = resp.json()
        check("cloud_api — /ml/predict has ml_risk_score",
              "ml_risk_score" in ml)
        check("cloud_api — /ml/predict score in [0,100]",
              0 <= ml["ml_risk_score"] <= 100,
              f"got {ml["ml_risk_score"]}")

    # ── GET /ml/model/status ──────────────────────────────────────
    resp = client.get("/ml/model/status")
    check("cloud_api — GET /ml/model/status returns 200",
          resp.status_code == 200)
    mstat = resp.json()
    check("cloud_api — model/status has local_model_exists field",
          "local_model_exists" in mstat)
    check("cloud_api — model/status has active_source field",
          "active_source" in mstat)

    # ── GET /recommendations/{id} ─────────────────────────────────
    resp = client.get("/recommendations/STU-001")
    check("cloud_api — GET /recommendations/STU-001 returns 200",
          resp.status_code == 200)
    rec = resp.json()
    check("cloud_api — recommendations list not empty",
          len(rec.get("recommendations", [])) > 0)
    check("cloud_api — recommendations have urgency field",
          all("urgency" in r for r in rec["recommendations"]))

    # ── CloudMLClient unit tests ──────────────────────────────────
    cloud_client = CloudMLClient()
    check("cloud_api — CloudMLClient not configured when no URL set",
          not cloud_client.is_configured)

    import asyncio
    ok = asyncio.get_event_loop().run_until_complete(cloud_client.health_check())
    check("cloud_api — CloudMLClient health_check returns False (no URL)",
          ok is False)

    result = asyncio.get_event_loop().run_until_complete(
        cloud_client.predict("STU-001", {})
    )
    check("cloud_api — CloudMLClient.predict returns None when not configured",
          result is None)

    # ── EngineRegistry lazy loading ───────────────────────────────
    reg = EngineRegistry()
    risk = reg.risk()
    check("cloud_api — EngineRegistry.risk() returns RiskEngine",
          risk is not None)
    rec2 = reg.rec()
    check("cloud_api — EngineRegistry.rec() returns RecommendationEngine",
          rec2 is not None)

    risk_block = sim["risk"]
    sim_block  = sim["simulation"]
    rule_score = risk_block["rule_based_score"]
    risk_lvl   = risk_block["risk_level"]
    base_sc    = sim_block["baseline_score"]
    proj_sc    = sim_block["projected_score"]
    n_recs     = len(sim["recommendations"])
    print(f"\n{INFO} /simulate response summary:")
    print(f"   Rule-based score : {rule_score}")
    print(f"   Risk level       : {risk_lvl}")
    print(f"   Baseline score   : {base_sc}")
    print(f"   Projected score  : {proj_sc}")
    print(f"   Recs generated   : {n_recs}")

def print_summary():
    passed = sum(_results)
    total  = len(_results)
    print(f"\n{'='*55}")
    if passed == total:
        print(f"\033[92m  ✓ All {total} tests passed.\033[0m")
    else:
        print(f"\033[91m  ✗ {passed}/{total} tests passed. "
              f"{total - passed} failed.\033[0m")
    print(f"{'='*55}\n")


def test_quantum_engine(*csv_paths):
    section("TEST F — quantum_engine.py")
    from quantum_engine import (
        QuantumRiskClassifier,
        QuantumInterventionOptimizer,
        get_quantum_risk_score,
    )
    from risk_engine import RiskEngine, StudentProfile, Trend

    student_data = {
        "absences": 18, "failures": 2,
        "G1": 5,        "G2": 4,
        "Dalc": 3,      "Walc": 4,
    }

    # ── Part A: QRC ───────────────────────────
    print(f"\n{INFO} Part A — Variational Quantum Classifier")

    if csv_paths and all(Path(p).exists() for p in csv_paths):
        qrc = QuantumRiskClassifier()
        qrc.train(*csv_paths, n_epochs=20, verbose=False)
        check("quantum — QRC training completes", qrc._is_trained)

        pred = qrc.predict("STU-001", student_data)
        check("quantum — QRC prediction in range [0,100]",
              0 <= pred.quantum_risk_score <= 100,
              f"got {pred.quantum_risk_score}")
        check("quantum — dropout_probability in [0,1]",
              0 <= pred.dropout_probability <= 1,
              f"got {pred.dropout_probability}")
        check("quantum — circuit_expectation in [-1,1]",
              -1 <= pred.circuit_expectation <= 1,
              f"got {pred.circuit_expectation}")
        check("quantum — confidence field set",
              pred.confidence in ("quantum-high", "quantum-low"))
        check("quantum — n_qubits correct",
              pred.n_qubits == 4)

        # Save / load
        qrc.save("test_qrc_weights.npy")
        qrc2  = QuantumRiskClassifier.load("test_qrc_weights.npy")
        pred2 = qrc2.predict("STU-001", student_data)
        check("quantum — save/load produces same score",
              abs(pred.quantum_risk_score - pred2.quantum_risk_score) < 0.5,
              f"{pred.quantum_risk_score} vs {pred2.quantum_risk_score}")

        # Three-way blend
        blended = get_quantum_risk_score(qrc, student_data, "STU-001",
                                          rule_based_score=82.0, ml_score=75.0)
        check("quantum — three-way blend in [1,99]",
              1 <= blended <= 99, f"got {blended}")
        print(f"\n{INFO} QRC score    : {pred.quantum_risk_score}")
        print(f"{INFO} Expectation  : {pred.circuit_expectation:+.4f}")
        print(f"{INFO} Blended score: {blended}")

    else:
        print(f"{INFO} No CSV provided — skipping QRC training tests.")
        print(f"{INFO} Run: python test_edusight.py quantum Maths.csv")

    # ── Part B: QIO ───────────────────────────
    print(f"\n{INFO} Part B — Quantum Intervention Optimizer")

    risk_engine = RiskEngine(ml_model_path=None)
    student     = StudentProfile(
        student_id="STU-001", name="Muhammad Ali bin Faisal",
        grade="Form 4", attendance_rate=62.0, academic_score=49.0,
        socio_score=55.0, family_support=40.0, trend=Trend.WORSENING,
    )

    qio    = QuantumInterventionOptimizer()
    result = qio.optimise(student, risk_engine, budget=40.0,
                           n_steps=20, verbose=False)

    check("quantum — QIO returns result",
          result is not None)
    check("quantum — QIO projected score < baseline",
          result.projected_risk_score < risk_engine.score(student).total_score,
          f"projected={result.projected_risk_score}")
    check("quantum — QIO risk_reduction > 0",
          result.risk_reduction > 0,
          f"reduction={result.risk_reduction}")
    check("quantum — attendance_boost in [0,30]",
          0 <= result.attendance_boost <= 30,
          f"got {result.attendance_boost}")
    check("quantum — academic_boost in [0,25]",
          0 <= result.academic_boost <= 25,
          f"got {result.academic_boost}")
    check("quantum — counselling_sessions in [0,10]",
          0 <= result.counselling_sessions <= 10,
          f"got {result.counselling_sessions}")
    check("quantum — welfare_support in {0, 0.5, 1}",
          result.welfare_support in (0.0, 0.5, 1.0),
          f"got {result.welfare_support}")
    check("quantum — method label correct",
          result.method == "quantum-variational")

    print(f"\n{INFO} Optimal intervention found:")
    print(f"   Attendance +{result.attendance_boost}%  |  "
          f"Academic +{result.academic_boost} pts  |  "
          f"Counselling {result.counselling_sessions}  |  "
          f"Welfare {result.welfare_support}")
    print(f"   Projected risk: {result.projected_risk_score}  "
          f"(−{result.risk_reduction} pts)")




if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    csv_paths = sys.argv[2:] if len(sys.argv) > 2 else []

    print("\n╔══════════════════════════════════════════════════╗")
    print("║       EduSight — Integration Test Runner        ║")
    print("╚══════════════════════════════════════════════════╝")

    if mode == "rule":
        test_risk_engine_rule_based()
        test_recommendation_engine()
        test_full_pipeline()

    elif mode == "ml":
        test_ml_engine(*csv_paths)
        test_ml_enhanced_pipeline()

    elif mode == "quantum":
        test_quantum_engine(*csv_paths)

    elif mode == "api":
        test_cloud_api()

    elif mode == "all":
        test_risk_engine_rule_based()
        test_recommendation_engine()
        test_full_pipeline()
        test_ml_enhanced_pipeline()
        test_quantum_engine(*csv_paths)
        test_cloud_api()

    else:
        print(f"Unknown mode '{mode}'. Use: rule | ml | all")
        sys.exit(1)

    print_summary()
