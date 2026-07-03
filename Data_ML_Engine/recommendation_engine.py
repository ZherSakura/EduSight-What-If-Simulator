"""
EduSight — Recommendation Engine
==================================
Generates prioritised, context-aware intervention recommendations from a student's RiskScore and SimulationResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ── import all shared types from risk_engine ─────────────────────────────────
from risk_engine import (
    RiskLevel,
    RiskScore,
    SimulationInput,
    SimulationResult,
    StudentProfile,
    Trend,
)


# ─────────────────────────────────────────────
#  Domain types
# ─────────────────────────────────────────────

class InterventionCategory(str, Enum):
    ATTENDANCE  = "attendance"
    ACADEMIC    = "academic"
    WELFARE     = "welfare"
    COUNSELLING = "counselling"
    FAMILY      = "family_engagement"
    MONITORING  = "monitoring"


class UrgencyLevel(str, Enum):
    CRITICAL = "critical"   # Act within days
    HIGH     = "high"       # Act within a week
    MEDIUM   = "medium"     # Act within a month
    ROUTINE  = "routine"    # Ongoing / preventive


@dataclass
class Recommendation:
    """A single actionable intervention card shown in the UI."""
    category                 : InterventionCategory
    urgency                  : UrgencyLevel
    title                    : str
    description              : str
    expected_impact          : str
    owner                    : str
    icon                     : str
    estimated_score_reduction: float = 0.0


@dataclass
class RecommendationReport:
    """Full output for one student — returned by POST /simulate."""
    student_id      : str
    student_name    : str
    risk_level      : RiskLevel
    recommendations : list
    narrative       : str   # paragraph in '3-month dropout probability' card
    priority_action : str   # top card headline


# ─────────────────────────────────────────────
#  Thresholds (tunable by ML team)
# ─────────────────────────────────────────────

ATTENDANCE_CRITICAL = 60.0
ATTENDANCE_LOW      = 75.0
ACADEMIC_CRITICAL   = 40.0
ACADEMIC_LOW        = 55.0
SOCIO_HIGH          = 60.0
FAMILY_LOW          = 45.0


# ─────────────────────────────────────────────
#  Engine
# ─────────────────────────────────────────────

class RecommendationEngine:
    """
    Rule-based recommendation engine with simulation-aware re-ranking.

    When a SimulationResult is supplied, the engine boosts recommendations
    whose intervention category drove the most score reduction.

    Usage (standalone)
    ------------------
    rec_engine = RecommendationEngine()
    report = rec_engine.recommend(student, risk_score)

    Usage (with simulation)
    -----------------------
    report = rec_engine.recommend(student, risk_score,
                                  sim_input=sim_input,
                                  sim_result=sim_result)

    Full pipeline (risk_engine + recommendation_engine together)
    ------------------------------------------------------------
    from risk_engine import RiskEngine, StudentProfile, SimulationInput, Trend
    from recommendation_engine import RecommendationEngine

    risk_engine = RiskEngine()           # or RiskEngine(ml_model_path="edusight_ml_model.joblib")
    rec_engine  = RecommendationEngine()

    student    = StudentProfile(...)
    risk_score = risk_engine.score(student)
    sim_result = risk_engine.simulate(student, SimulationInput(...))
    report     = rec_engine.recommend(student, risk_score,
                                      sim_input=SimulationInput(...),
                                      sim_result=sim_result)
    """

    def recommend(
        self,
        student   : StudentProfile,
        risk_score: RiskScore,
        sim_input : Optional[SimulationInput]  = None,
        sim_result: Optional[SimulationResult] = None,
    ) -> RecommendationReport:

        recs = []
        recs.extend(self._attendance_rules(student, risk_score))
        recs.extend(self._academic_rules(student, risk_score))
        recs.extend(self._welfare_rules(student, risk_score))
        recs.extend(self._counselling_rules(student, risk_score))
        recs.extend(self._family_rules(student, risk_score))
        recs.extend(self._monitoring_rules(student, risk_score))

        if sim_result:
            recs = self._rerank_with_simulation(recs, sim_input, sim_result)

        recs.sort(key=lambda r: (self._urgency_order(r.urgency),
                                  -r.estimated_score_reduction))
        recs = self._deduplicate(recs)

        narrative       = self._build_narrative(student, risk_score, sim_result)
        priority_action = recs[0].description if recs else "Continue monitoring student progress."

        return RecommendationReport(
            student_id      = student.student_id,
            student_name    = student.name,
            risk_level      = risk_score.risk_level,
            recommendations = recs,
            narrative       = narrative,
            priority_action = priority_action,
        )

    # ── rule sets ───────────────────────────

    def _attendance_rules(self, s, rs) -> list:
        recs = []
        if s.attendance_rate < ATTENDANCE_CRITICAL:
            recs.append(Recommendation(
                category=InterventionCategory.ATTENDANCE, urgency=UrgencyLevel.CRITICAL,
                title="Immediate parent/guardian contact",
                description=(
                    "Increase attendance to at least 80% to see meaningful risk reduction. "
                    "Contact parent/guardian within 48 hours."
                ),
                expected_impact="reduce risk score by ~12–15 pts",
                owner="Homeroom Teacher", icon="user-check",
                estimated_score_reduction=13.0,
            ))
        elif s.attendance_rate < ATTENDANCE_LOW:
            recs.append(Recommendation(
                category=InterventionCategory.ATTENDANCE, urgency=UrgencyLevel.HIGH,
                title="Schedule PIBG engagement meeting",
                description=(
                    "Attendance below 75%. Schedule parent–teacher meeting "
                    "through PIBG to identify barriers to attendance."
                ),
                expected_impact="reduce risk score by ~7–10 pts",
                owner="Homeroom Teacher + PIBG", icon="users",
                estimated_score_reduction=8.5,
            ))
        if s.trend == Trend.WORSENING and s.attendance_rate < 80:
            recs.append(Recommendation(
                category=InterventionCategory.ATTENDANCE, urgency=UrgencyLevel.HIGH,
                title="Attendance Improvement Plan (AIP)",
                description=(
                    "Worsening attendance trend detected. Create a formal "
                    "Attendance Improvement Plan with weekly check-ins."
                ),
                expected_impact="stabilise trend, reduce risk by ~5 pts",
                owner="Discipline Teacher", icon="clipboard-list",
                estimated_score_reduction=5.0,
            ))
        return recs

    def _academic_rules(self, s, rs) -> list:
        recs = []
        if s.academic_score < ACADEMIC_CRITICAL:
            recs.append(Recommendation(
                category=InterventionCategory.ACADEMIC, urgency=UrgencyLevel.CRITICAL,
                title="Enrol in intensive remedial programme",
                description=(
                    "Academic tutoring needed — grade below passing threshold. "
                    "Enrol in after-school remedial classes for core subjects immediately."
                ),
                expected_impact="reduce risk score by ~8–12 pts over 6 weeks",
                owner="Subject Teachers", icon="book-open",
                estimated_score_reduction=10.0,
            ))
        elif s.academic_score < ACADEMIC_LOW:
            recs.append(Recommendation(
                category=InterventionCategory.ACADEMIC, urgency=UrgencyLevel.HIGH,
                title="Peer tutoring / additional support",
                description=(
                    "Academic score below passing benchmark. "
                    "Pair with a peer tutor or assign supplemental exercises."
                ),
                expected_impact="reduce risk score by ~5–8 pts",
                owner="Subject Teachers", icon="book",
                estimated_score_reduction=6.0,
            ))
        return recs

    def _welfare_rules(self, s, rs) -> list:
        recs = []
        if s.socio_score >= SOCIO_HIGH:
            recs.append(Recommendation(
                category=InterventionCategory.WELFARE, urgency=UrgencyLevel.HIGH,
                title="Welfare assistance referral (RMT / BAP)",
                description=(
                    "High socioeconomic risk detected. Refer to school welfare "
                    "officer for RMT meal programme and BAP book assistance."
                ),
                expected_impact="reduce socioeconomic burden, risk score by ~6–8 pts",
                owner="School Welfare Officer", icon="heart-handshake",
                estimated_score_reduction=7.0,
            ))
        if s.socio_score >= 75:
            recs.append(Recommendation(
                category=InterventionCategory.WELFARE, urgency=UrgencyLevel.HIGH,
                title="KWAPM / SPBT application",
                description=(
                    "Severe socioeconomic hardship indicated. "
                    "Expedite KWAPM financial aid and SPBT textbook loan application."
                ),
                expected_impact="reduce dropout probability by ~10%",
                owner="Admin / Welfare Officer", icon="currency-dollar",
                estimated_score_reduction=9.0,
            ))
        return recs

    def _counselling_rules(self, s, rs) -> list:
        recs = []
        if rs.risk_level == RiskLevel.HIGH:
            recs.append(Recommendation(
                category=InterventionCategory.COUNSELLING, urgency=UrgencyLevel.HIGH,
                title="Begin structured counselling programme",
                description=(
                    "Schedule at least 3 counselling sessions to reduce risk "
                    "by ~8 points. Focus on motivation and personal barriers."
                ),
                expected_impact="reduce risk score by ~8 pts (3 sessions)",
                owner="School Counsellor", icon="mood-smile",
                estimated_score_reduction=8.0,
            ))
        elif rs.risk_level == RiskLevel.MEDIUM:
            recs.append(Recommendation(
                category=InterventionCategory.COUNSELLING, urgency=UrgencyLevel.MEDIUM,
                title="Bi-weekly check-in with counsellor",
                description=(
                    "Medium-risk student. Bi-weekly counsellor check-ins to build resilience."
                ),
                expected_impact="stabilise risk, prevent escalation to high",
                owner="School Counsellor", icon="mood-smile",
                estimated_score_reduction=4.0,
            ))
        return recs

    def _family_rules(self, s, rs) -> list:
        if s.family_support < FAMILY_LOW:
            return [Recommendation(
                category=InterventionCategory.FAMILY, urgency=UrgencyLevel.MEDIUM,
                title="Family engagement & home visit",
                description=(
                    "Low family support index detected. "
                    "Arrange home visit and involve parent in monthly progress review."
                ),
                expected_impact="improve family support score, reduce risk by ~5 pts",
                owner="Homeroom Teacher + Counsellor", icon="home",
                estimated_score_reduction=5.0,
            )]
        return []

    def _monitoring_rules(self, s, rs) -> list:
        return [Recommendation(
            category=InterventionCategory.MONITORING, urgency=UrgencyLevel.ROUTINE,
            title="Monthly risk review",
            description=(
                "Track attendance and academic score monthly. "
                "Re-run risk simulation if any factor drops by >10 points."
            ),
            expected_impact="early detection of risk escalation",
            owner="Homeroom Teacher", icon="chart-line",
            estimated_score_reduction=0.0,
        )]

    # ── simulation-aware re-ranking ─────────

    def _rerank_with_simulation(self, recs, sim_input, sim_result) -> list:
        if not sim_input or sim_result.score_delta >= 0:
            return recs
        boosts = {}
        if sim_input.attendance_boost     > 0:
            boosts[InterventionCategory.ATTENDANCE]  = sim_input.attendance_boost * 0.4
        if sim_input.academic_boost       > 0:
            boosts[InterventionCategory.ACADEMIC]    = sim_input.academic_boost * 0.3
        if sim_input.counselling_sessions > 0:
            boosts[InterventionCategory.COUNSELLING] = sim_input.counselling_sessions * 1.5
        if sim_input.welfare_support      > 0:
            boosts[InterventionCategory.WELFARE]     = sim_input.welfare_support * 8
        for rec in recs:
            rec.estimated_score_reduction += boosts.get(rec.category, 0.0)
        return recs

    # ── helpers ─────────────────────────────

    @staticmethod
    def _urgency_order(u) -> int:
        return {
            UrgencyLevel.CRITICAL: 0, UrgencyLevel.HIGH: 1,
            UrgencyLevel.MEDIUM: 2,   UrgencyLevel.ROUTINE: 3,
        }[u]

    @staticmethod
    def _deduplicate(recs) -> list:
        seen, unique = set(), []
        for r in recs:
            if r.category not in seen:
                seen.add(r.category)
                unique.append(r)
        return unique

    @staticmethod
    def _build_narrative(student, risk_score, sim_result) -> str:
        prob_pct = round(risk_score.dropout_prob_3m * 100)
        opener   = {
            Trend.WORSENING: "At the ongoing trajectory",
            Trend.STABLE   : "If the current situation continues",
            Trend.IMPROVING: "With the improvements observed",
        }[risk_score.trend]
        text = (
            f"{opener}, {student.name} is at {risk_score.risk_level.value} risk "
            f"of dropping out within 3 months ({prob_pct}% probability). "
            f"Immediate action is recommended."
        )
        if sim_result and sim_result.score_delta < 0:
            reduction = abs(sim_result.score_delta)
            new_prob  = round(sim_result.projected_prob_3m * 100)
            text += (
                f" Simulated interventions could reduce the risk score by "
                f"{reduction:.0f} points, bringing 3-month dropout probability "
                f"down to {new_prob}%."
            )
        return text


# ─────────────────────────────────────────────
#  Smoke-test  (python recommendation_engine.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from risk_engine import RiskEngine, StudentProfile, SimulationInput, Trend

    risk_engine = RiskEngine(ml_model_path=None)   # pure rule-based for quick test
    rec_engine  = RecommendationEngine()

    student = StudentProfile(
        student_id="STU-001", name="Muhammad Ali bin Faisal",
        grade="Form 4", attendance_rate=62.0, academic_score=49.0,
        socio_score=55.0, family_support=40.0, trend=Trend.WORSENING,
    )

    risk_score = risk_engine.score(student)
    sim_input  = SimulationInput(attendance_boost=18.0, academic_boost=10.0,
                                  counselling_sessions=5, welfare_support=1.0)
    sim_result = risk_engine.simulate(student, sim_input)
    report     = rec_engine.recommend(student, risk_score,
                                       sim_input=sim_input, sim_result=sim_result)

    print("=== Recommendation Report ===")
    print(f"Student   : {report.student_name}")
    print(f"Risk Level: {report.risk_level.value}")
    print(f"Narrative : {report.narrative}")
    print(f"\nTop Priority: {report.priority_action}")
    print(f"\n--- Recommendations ({len(report.recommendations)}) ---")
    for i, r in enumerate(report.recommendations, 1):
        print(f"{i}. [{r.urgency.value.upper()}] {r.title}")
        print(f"   {r.description}")
        print(f"   Impact: {r.expected_impact} | Owner: {r.owner}\n")
