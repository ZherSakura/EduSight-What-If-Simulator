from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


# ── lazy ML import so risk_engine works even without a trained model ──────────
def _try_load_ml(model_path: Optional[str]):
    """
    Attempt to load a trained MLEngine from disk.
    Returns None silently if the model file does not exist yet,
    so the rule-based engine still works standalone.
    """
    if model_path is None:
        return None
    try:
        from ml_engine import MLEngine
        if Path(model_path).exists():
            return MLEngine.load(model_path)
        else:
            print(f"[RiskEngine] ML model not found at '{model_path}'. "
                  f"Running in rule-based mode. "
                  f"Train & save a model first: MLEngine().train('Maths.csv').save()")
            return None
    except ImportError:
        print("[RiskEngine] ml_engine.py not found. Running in rule-based mode.")
        return None


# ─────────────────────────────────────────────
#  Domain types
# ─────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW    = "low"       # 0–39
    MEDIUM = "medium"    # 40–64
    HIGH   = "high"      # 65–100


class Trend(str, Enum):
    IMPROVING  = "improving"
    STABLE     = "stable"
    WORSENING  = "worsening"


# ─────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────

@dataclass
class StudentProfile:
    """
    Core student data consumed by the risk engine.

    Fields
    ------
    student_id      : unique identifier (maps to GET /students/:id)
    name            : display name
    grade           : e.g. "Form 4"
    attendance_rate : percentage 0–100  (from e-Kehadiran)
    academic_score  : percentage 0–100  (from PBD / UASA)
    socio_score     : 0–100; higher = more disadvantaged
    family_support  : 0–100; higher = stronger support
    trend           : historical trajectory over last 3 months
    ml_data         : optional raw CSV-column dict for ML inference
                      e.g. {"G1": 5, "G2": 4, "absences": 18, ...}
    """
    student_id     : str
    name           : str
    grade          : str
    attendance_rate: float
    academic_score : float
    socio_score    : float
    family_support : float = 60.0
    trend          : Trend = Trend.STABLE
    ml_data        : Optional[dict] = None   # raw feature dict for MLEngine


@dataclass
class SimulationInput:
    """Intervention adjustments from the What-If Simulator sliders."""
    attendance_boost    : float = 0.0
    academic_boost      : float = 0.0
    counselling_sessions: int   = 0
    welfare_support     : float = 0.0   # 0.0 / 0.5 / 1.0


@dataclass
class RiskScore:
    """Computed risk profile for a single student."""
    student_id      : str
    total_score     : float
    risk_level      : RiskLevel
    factor_weights  : dict
    dropout_prob_3m : float
    dropout_prob_6m : float
    trend           : Trend
    explanation     : str
    ml_enhanced     : bool = False   # True if ML blend was applied


@dataclass
class SimulationResult:
    """Returned by simulate()."""
    baseline_score      : float
    projected_score     : float
    score_delta         : float          # negative = improvement
    baseline_prob_3m    : float
    projected_prob_3m   : float
    risk_level_baseline : RiskLevel
    risk_level_projected: RiskLevel
    factor_weights      : dict
    dominant_factor     : str
    summary             : str


# ─────────────────────────────────────────────
#  Weight configuration
# ─────────────────────────────────────────────

DEFAULT_WEIGHTS: dict = {
    "attendance"          : 0.45,
    "academic_performance": 0.30,
    "socioeconomic_status": 0.13,
    "family_support"      : 0.12,
}

# Blend weight: how much ML score overrides rule-based score (0–1).
# Set to 0.0 to use pure rule-based; 1.0 for pure ML.
ML_BLEND_WEIGHT: float = 0.40


# ─────────────────────────────────────────────
#  Core engine
# ─────────────────────────────────────────────

class RiskEngine:
    """
    Computes dropout risk scores and What-If simulations.

    ML integration
    --------------
    Pass ml_model_path to activate ML blending:

        engine = RiskEngine(ml_model_path="edusight_ml_model.joblib")

    When a trained model is found AND the student has ml_data populated,
    the final score is:
        score = (1 - ML_BLEND_WEIGHT) × rule_score
              +      ML_BLEND_WEIGHT  × ml_score

    If no model file exists, engine runs in pure rule-based mode.
    """

    def __init__(
        self,
        weights       : Optional[dict] = None,
        ml_model_path : Optional[str]  = "edusight_ml_model.joblib",
    ):
        self.weights = weights or DEFAULT_WEIGHTS
        self._validate_weights()
        self.ml = _try_load_ml(ml_model_path)

    # ── public ──────────────────────────────

    def score(self, student: StudentProfile) -> RiskScore:
        """Compute the current dropout risk score for a student."""
        raw            = self._raw_factors(student)
        rule_total     = self._weighted_total(raw)
        factor_weights = self._factor_percentages(raw)

        # ── ML blend (only if model loaded AND student has raw feature dict) ──
        final_total = rule_total
        ml_enhanced = False
        if self.ml is not None and student.ml_data:
            try:
                from ml_engine import ml_risk_to_student_profile_override
                ml_pred     = self.ml.predict_risk(student.student_id, student.ml_data)
                final_total = ml_risk_to_student_profile_override(
                    ml_pred, rule_total, blend_weight=ML_BLEND_WEIGHT
                )
                ml_enhanced = True
            except Exception as e:
                print(f"[RiskEngine] ML inference failed for {student.student_id}: {e}. "
                      f"Falling back to rule-based score.")
                final_total = rule_total

        final_total = round(max(1.0, min(99.0, final_total)), 1)
        level       = self._level(final_total)
        prob_3m     = self._dropout_probability(final_total, student.trend, months=3)
        prob_6m     = self._dropout_probability(final_total, student.trend, months=6)
        explanation = self._explain(student.name, final_total, level, factor_weights,
                                    student.trend, ml_enhanced)

        return RiskScore(
            student_id      = student.student_id,
            total_score     = final_total,
            risk_level      = level,
            factor_weights  = factor_weights,
            dropout_prob_3m = prob_3m,
            dropout_prob_6m = prob_6m,
            trend           = student.trend,
            explanation     = explanation,
            ml_enhanced     = ml_enhanced,
        )

    def simulate(self, student: StudentProfile, inputs: SimulationInput) -> SimulationResult:
        """Apply intervention deltas and return the projected risk."""
        baseline_risk  = self.score(student)
        baseline_score = baseline_risk.total_score
        baseline_3m    = baseline_risk.dropout_prob_3m

        modified = StudentProfile(
            student_id      = student.student_id,
            name            = student.name,
            grade           = student.grade,
            attendance_rate = min(100.0, student.attendance_rate
                                  + inputs.attendance_boost
                                  + inputs.counselling_sessions * 0.8),
            academic_score  = min(100.0, student.academic_score
                                  + inputs.academic_boost
                                  + inputs.welfare_support * 4),
            socio_score     = max(0.0,   student.socio_score
                                  - inputs.welfare_support * 18),
            family_support  = min(100.0, student.family_support
                                  + inputs.counselling_sessions * 3),
            trend           = student.trend,
            ml_data         = None,   # simulation uses rule-based only (sliders are hypothetical)
        )

        projected_risk  = self.score(modified)
        projected_score = projected_risk.total_score
        projected_3m    = projected_risk.dropout_prob_3m
        delta           = projected_score - baseline_score
        dominant        = max(projected_risk.factor_weights,
                              key=projected_risk.factor_weights.get)

        return SimulationResult(
            baseline_score       = round(baseline_score, 1),
            projected_score      = round(projected_score, 1),
            score_delta          = round(delta, 1),
            baseline_prob_3m     = round(baseline_3m, 3),
            projected_prob_3m    = round(projected_3m, 3),
            risk_level_baseline  = self._level(baseline_score),
            risk_level_projected = self._level(projected_score),
            factor_weights       = projected_risk.factor_weights,
            dominant_factor      = dominant,
            summary              = self._sim_summary(
                student.name, baseline_score, projected_score,
                baseline_3m, projected_3m, inputs),
        )

    # ── private helpers ──────────────────────

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0; got {total:.3f}.")

    def _raw_factors(self, s: StudentProfile) -> dict:
        return {
            "attendance"          : 100.0 - s.attendance_rate,
            "academic_performance": 100.0 - s.academic_score,
            "socioeconomic_status": s.socio_score,
            "family_support"      : 100.0 - s.family_support,
        }

    def _weighted_total(self, raw: dict) -> float:
        return round(max(1.0, min(99.0,
            sum(raw[k] * self.weights[k] for k in self.weights))), 1)

    def _factor_percentages(self, raw: dict) -> dict:
        weighted = {k: raw[k] * self.weights[k] for k in self.weights}
        total    = sum(weighted.values()) or 1.0
        return {k: round((v / total) * 100, 1) for k, v in weighted.items()}

    def _dropout_probability(self, score: float, trend: Trend, months: int) -> float:
        trend_shift = {"worsening": 8, "stable": 0, "improving": -8}[trend]
        adjusted    = score + (months / 3) * trend_shift
        prob        = 1.0 / (1.0 + math.exp(-(adjusted - 50) / 12))
        return round(max(0.01, min(0.99, prob)), 3)

    @staticmethod
    def _level(score: float) -> RiskLevel:
        if score >= 65: return RiskLevel.HIGH
        if score >= 40: return RiskLevel.MEDIUM
        return RiskLevel.LOW

    @staticmethod
    def _explain(name, score, level, factors, trend, ml_enhanced) -> str:
        dominant  = max(factors, key=factors.get)
        label_map = {
            "attendance"          : "poor attendance",
            "academic_performance": "low academic performance",
            "socioeconomic_status": "socioeconomic hardship",
            "family_support"      : "limited family support",
        }
        suffix = " (ML-enhanced)" if ml_enhanced else " (rule-based)"
        return (
            f"{name} has a risk score of {score:.0f} ({level.value} risk). "
            f"Primary driver: {label_map[dominant]} "
            f"({factors[dominant]:.0f}% of weight). "
            f"Trend: {trend.value}.{suffix}"
        )

    @staticmethod
    def _sim_summary(name, baseline, projected, base_3m, proj_3m, inputs) -> str:
        delta   = baseline - projected
        actions = []
        if inputs.attendance_boost     > 0: actions.append("attendance improvement")
        if inputs.academic_boost       > 0: actions.append("academic tutoring")
        if inputs.counselling_sessions > 0: actions.append("counselling")
        if inputs.welfare_support      > 0: actions.append("welfare support")
        if delta <= 0:
            return f"No interventions applied. {name}'s projected risk remains at {projected:.0f}."
        action_str = ", ".join(actions) if actions else "combined interventions"
        pct        = (delta / baseline * 100) if baseline else 0
        return (
            f"With {action_str}, {name}'s risk score drops from "
            f"{baseline:.0f} → {projected:.0f} (−{pct:.0f}% reduction). "
            f"3-month dropout probability: {base_3m*100:.0f}% → {proj_3m*100:.0f}%."
        )


# ─────────────────────────────────────────────
#  Smoke-test  (python risk_engine.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Rule-based mode (no ML model needed)
    engine = RiskEngine(ml_model_path=None)

    student = StudentProfile(
        student_id      = "STU-001",
        name            = "Muhammad Ali bin Faisal",
        grade           = "Form 4",
        attendance_rate = 62.0,
        academic_score  = 49.0,
        socio_score     = 55.0,
        family_support  = 40.0,
        trend           = Trend.WORSENING,
    )

    score = engine.score(student)
    print("=== Baseline Risk Score ===")
    print(f"Total Score    : {score.total_score}")
    print(f"Risk Level     : {score.risk_level.value}")
    print(f"Factor Weights : {score.factor_weights}")
    print(f"Prob 3m        : {score.dropout_prob_3m * 100:.1f}%")
    print(f"Prob 6m        : {score.dropout_prob_6m * 100:.1f}%")
    print(f"ML Enhanced    : {score.ml_enhanced}")
    print(f"Explanation    : {score.explanation}")

    result = engine.simulate(student, SimulationInput(
        attendance_boost=18.0, academic_boost=10.0,
        counselling_sessions=5, welfare_support=1.0,
    ))
    print("\n=== Simulation Result ===")
    print(f"Baseline  : {result.baseline_score}")
    print(f"Projected : {result.projected_score}")
    print(f"Delta     : {result.score_delta}")
    print(f"Summary   : {result.summary}")
