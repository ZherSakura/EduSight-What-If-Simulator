"""
EduSight — Quantum Engine
==========================
Adds quantum computing capabilities to the What-If Simulator via:
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Quantum circuit configuration
# ─────────────────────────────────────────────

# Classifier: 4 qubits (one per input feature), 3 variational layers
N_QUBITS_CLASSIFIER = 4
N_LAYERS_CLASSIFIER = 3

# Optimizer: 4 qubits (one per intervention slider), 2 layers
N_QUBITS_OPTIMIZER  = 4
N_LAYERS_OPTIMIZER  = 2

# The 4 features the VQC encodes (normalisable to [0, π])
QUANTUM_FEATURES = [
    "absences",            # 0–93  → strongest dropout signal
    "failures",            # 0–4   → past academic failure count
    "feat_avg_prior_grade",# 0–20  → average of G1 + G2
    "feat_alcohol_risk",   # 0–5   → derived from Dalc + Walc
]

DEFAULT_QRC_PATH = Path("edusight_qrc_weights.npy")


# ─────────────────────────────────────────────
#  Output schemas
# ─────────────────────────────────────────────

@dataclass
class QuantumRiskPrediction:
    """Result from the Variational Quantum Classifier."""
    student_id          : str
    quantum_risk_score  : float   # 0–100
    dropout_probability : float   # 0–1
    circuit_expectation : float   # raw PauliZ expectation (-1 to +1)
    n_qubits            : int
    n_layers            : int
    confidence          : str     # "quantum-high" / "quantum-low"


@dataclass
class QuantumOptimisationResult:
    """Best intervention combination found by the Quantum Optimizer."""
    attendance_boost    : float
    academic_boost      : float
    counselling_sessions: int
    welfare_support     : float
    projected_risk_score: float
    risk_reduction      : float   # baseline_score - projected_score
    optimisation_steps  : int
    method              : str     # always "quantum-variational"


# ─────────────────────────────────────────────
#  Feature normaliser
# ─────────────────────────────────────────────

class QuantumFeatureNormaliser:
    """
    Normalises raw feature values to [0, π] for angle embedding.
    Must be fit on training data; reused for inference.
    """

    def __init__(self):
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "QuantumFeatureNormaliser":
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._min is None:
            raise RuntimeError("Call fit() before transform().")
        rng = np.where(self._max - self._min > 0, self._max - self._min, 1.0)
        return ((X - self._min) / rng) * np.pi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────
#  Component 1 — Variational Quantum Classifier
# ─────────────────────────────────────────────

class QuantumRiskClassifier:
    """
    Variational Quantum Classifier (VQC) for student dropout risk.

    Circuit architecture
    ────────────────────
    1. AngleEmbedding  : encodes 4 normalised features as RY rotations
    2. BasicEntanglerLayers : RX rotations + CNOT ring (N_LAYERS_CLASSIFIER)
    3. Measurement     : PauliZ expectation on qubit 0 → maps to risk score

    Training
    ────────
    Labels  : +1 = at risk (G3 < 10),  -1 = not at risk
    Loss    : MSE between PauliZ expectation and ±1 label
    Optimiser: Gradient descent with parameter-shift rule (hardware-compatible)

    Usage
    ─────
    # Train fresh
    qrc = QuantumRiskClassifier()
    qrc.train("Maths.csv", n_epochs=40)
    qrc.save()

    # Or load saved weights
    qrc = QuantumRiskClassifier.load()

    # Inference
    pred = qrc.predict("STU-001", student_data_dict)
    """

    def __init__(self):
        self._dev        = qml.device("default.qubit", wires=N_QUBITS_CLASSIFIER)
        self._weights    : Optional[pnp.ndarray] = None
        self._normaliser = QuantumFeatureNormaliser()
        self._is_trained = False
        self._circuit    = self._build_circuit()

    # ── public ──────────────────────────────

    def train(
        self,
        *csv_paths   : str | Path,
        n_epochs     : int   = 40,
        learning_rate: float = 0.3,
        batch_size   : int   = 16,
        verbose      : bool  = True,
    ) -> "QuantumRiskClassifier":
        """Train the VQC on student CSV data."""
        if verbose:
            print(f"[QRC] Quantum circuit: "
                  f"{N_QUBITS_CLASSIFIER} qubits × {N_LAYERS_CLASSIFIER} layers")

        X, y = self._load_data(*csv_paths)
        X_norm = self._normaliser.fit_transform(X)

        self._weights = pnp.random.uniform(
            0, np.pi,
            (N_LAYERS_CLASSIFIER, N_QUBITS_CLASSIFIER),
            requires_grad=True,
        )

        opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

        if verbose:
            print(f"[QRC] Training on {len(X)} samples, {n_epochs} epochs...")

        for epoch in range(n_epochs):
            idx     = np.random.choice(len(X_norm),
                                        min(batch_size, len(X_norm)), replace=False)
            X_batch = X_norm[idx]
            y_batch = y[idx]

            def cost(w):
                preds = pnp.array([
                    self._circuit(pnp.array(x, requires_grad=False), w)
                    for x in X_batch
                ])
                return pnp.mean(
                    (preds - pnp.array(y_batch, requires_grad=False)) ** 2
                )

            self._weights, loss = opt.step_and_cost(cost, self._weights)

            if verbose and (epoch + 1) % 10 == 0:
                acc = self._compute_accuracy(X_norm, y)
                print(f"[QRC]   Epoch {epoch+1:>3}/{n_epochs}  "
                      f"loss={loss:.4f}  acc={acc:.3f}")

        self._is_trained = True
        if verbose:
            print(f"[QRC] Training complete. "
                  f"Final acc: {self._compute_accuracy(X_norm, y):.3f}")
        return self

    def predict(self, student_id: str, student_data: dict) -> QuantumRiskPrediction:
        """
        Predict dropout risk for a single student.

        student_data must include the keys in QUANTUM_FEATURES, or the
        raw CSV columns needed to derive them
        (absences, failures, G1, G2, Dalc, Walc).
        """
        self._assert_trained()

        x_raw  = self._extract_features(student_data)
        x_norm = self._normaliser.transform(x_raw.reshape(1, -1))[0]
        x_pnp  = pnp.array(x_norm, requires_grad=False)

        expval = float(self._circuit(x_pnp, self._weights))

        # Map (-1, +1) → (0, 100)
        quantum_risk = round((1 - expval) / 2 * 100, 1)
        dropout_prob = round((1 - expval) / 2, 3)
        confidence   = "quantum-high" if abs(expval) > 0.3 else "quantum-low"

        return QuantumRiskPrediction(
            student_id          = student_id,
            quantum_risk_score  = quantum_risk,
            dropout_probability = dropout_prob,
            circuit_expectation = round(expval, 4),
            n_qubits            = N_QUBITS_CLASSIFIER,
            n_layers            = N_LAYERS_CLASSIFIER,
            confidence          = confidence,
        )

    def save(self, path: str | Path = DEFAULT_QRC_PATH) -> None:
        """Save trained weights and normaliser to disk."""
        self._assert_trained()
        path = Path(path)
        np.save(str(path), np.array(self._weights))
        np.save(str(path).replace(".npy", "_norm_min.npy"), self._normaliser._min)
        np.save(str(path).replace(".npy", "_norm_max.npy"), self._normaliser._max)
        print(f"[QRC] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_QRC_PATH) -> "QuantumRiskClassifier":
        """Load previously trained weights from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"QRC weights not found: {path}\n"
                f"Train first: QuantumRiskClassifier().train('Maths.csv').save()"
            )
        qrc = cls()
        qrc._weights             = pnp.array(np.load(str(path)), requires_grad=True)
        qrc._normaliser._min     = np.load(str(path).replace(".npy", "_norm_min.npy"))
        qrc._normaliser._max     = np.load(str(path).replace(".npy", "_norm_max.npy"))
        qrc._is_trained          = True
        print(f"[QRC] Loaded ← {path}")
        return qrc

    # ── private ──────────────────────────────

    def _build_circuit(self):
        dev = self._dev

        @qml.qnode(dev)
        def circuit(inputs: pnp.ndarray, weights: pnp.ndarray) -> float:
            qml.AngleEmbedding(inputs, wires=range(N_QUBITS_CLASSIFIER), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS_CLASSIFIER))
            return qml.expval(qml.PauliZ(0))

        return circuit

    def _load_data(self, *csv_paths: str | Path):
        try:
            from ml_engine import StudentDataLoader
        except ImportError:
            raise ImportError("ml_engine.py must be in the same folder.")

        loader = StudentDataLoader()
        df     = loader.load(*csv_paths)

        if "feat_avg_prior_grade" not in df.columns and "G1" in df.columns:
            df["feat_avg_prior_grade"] = (df["G1"] + df["G2"]) / 2
        if "feat_alcohol_risk" not in df.columns and "Dalc" in df.columns:
            df["feat_alcohol_risk"] = (df["Dalc"] * 2 + df["Walc"]) / 3

        for feat in QUANTUM_FEATURES:
            if feat not in df.columns:
                df[feat] = 0.0

        X = df[QUANTUM_FEATURES].fillna(0).values.astype(float)
        y = np.where(df["label_atrisk"].values == 1, 1.0, -1.0)
        return X, y

    def _compute_accuracy(self, X_norm: np.ndarray, y: np.ndarray) -> float:
        preds = np.array([
            float(self._circuit(pnp.array(x, requires_grad=False), self._weights))
            for x in X_norm
        ])
        return float(np.mean(np.where(preds >= 0, 1.0, -1.0) == y))

    @staticmethod
    def _extract_features(d: dict) -> np.ndarray:
        absences = float(d.get("absences", 5))
        failures = float(d.get("failures", 0))
        g1       = float(d.get("G1", d.get("feat_avg_prior_grade", 10)))
        g2       = float(d.get("G2", g1))
        avg_g    = float(d.get("feat_avg_prior_grade", (g1 + g2) / 2))
        dalc     = float(d.get("Dalc", 1))
        walc     = float(d.get("Walc", 1))
        alc      = float(d.get("feat_alcohol_risk", (dalc * 2 + walc) / 3))
        return np.array([absences, failures, avg_g, alc])

    def _assert_trained(self) -> None:
        if not self._is_trained or self._weights is None:
            raise RuntimeError(
                "QRC not trained. Call .train('Maths.csv') "
                "or load weights with QuantumRiskClassifier.load()."
            )


# ─────────────────────────────────────────────
#  Component 2 — Quantum Intervention Optimizer
# ─────────────────────────────────────────────

class QuantumInterventionOptimizer:
    """
    QAOA-inspired variational optimizer for the What-If Simulator.

    Design
    ──────
    4 qubits — one per intervention slider.  The circuit's variational
    parameters are trained to output qubit expectation values that decode
    into the best (attendance_boost, academic_boost, counselling_sessions,
    welfare_support) combination.

    At each optimisation step:
      1. Run the quantum circuit → 4 PauliZ expectation values ∈ (-1, +1)
      2. Decode expvals → slider values within the budget
      3. Call RiskEngine.simulate() outside autograd to get projected_score
      4. Quantum cost function = weighted sum of expvals (differentiable proxy)
         The proxy is calibrated so maximising intervention magnitudes
         correlates with minimising projected risk
      5. Update variational parameters via AdamOptimizer (parameter-shift)
      6. Track the best projected_score seen across all steps

    This two-step approach (quantum circuit differentiable, RiskEngine
    evaluated outside autograd) is the standard hybrid quantum-classical
    pattern used in production quantum ML pipelines.

    Budget encoding
    ───────────────
      attendance_boost    : 1 unit = 1 pp  (range 0–30)
      academic_boost      : 1 unit = 1 pt  (range 0–25)
      counselling_sessions: 1 unit = 1 session (range 0–10)
      welfare_support     : 10 units per level {0, 0.5, 1}
      Default budget = 40 → e.g. 15pp + 10pts + 5 sessions + 0.5 welfare

    Usage
    ─────
    qio    = QuantumInterventionOptimizer()
    result = qio.optimise(student, risk_engine, budget=40.0, n_steps=30)
    print(result.attendance_boost, result.academic_boost, ...)
    """

    def __init__(self):
        self._dev = qml.device("default.qubit", wires=N_QUBITS_OPTIMIZER)
        self._circuit = self._build_circuit()

    # ── public ──────────────────────────────

    def optimise(
        self,
        student      ,           # StudentProfile from risk_engine.py
        risk_engine  ,           # RiskEngine instance
        budget       : float = 40.0,
        n_steps      : int   = 30,
        learning_rate: float = 0.3,
        verbose      : bool  = True,
    ) -> QuantumOptimisationResult:
        """
        Find the best slider combination within the resource budget.

        Returns QuantumOptimisationResult with all 4 slider values,
        projected risk score, and total risk reduction achieved.
        """
        from risk_engine import SimulationInput

        baseline_score = risk_engine.score(student).total_score

        if verbose:
            print(f"[QIO] Student: {student.name}")
            print(f"[QIO] Baseline: {baseline_score:.1f}  Budget: {budget}")
            print(f"[QIO] Circuit: {N_QUBITS_OPTIMIZER}q × {N_LAYERS_OPTIMIZER}L  "
                  f"Steps: {n_steps}")

        params = pnp.random.uniform(
            0, np.pi,
            (N_LAYERS_OPTIMIZER, N_QUBITS_OPTIMIZER, 2),
            requires_grad=True,
        )

        opt         = qml.AdamOptimizer(stepsize=learning_rate)
        best_score  = baseline_score
        best_params = pnp.array(params)

        for step in range(n_steps):
            # ── Step 1: evaluate current params with RiskEngine ──────────
            # (outside autograd so RiskEngine's Python logic doesn't block grads)
            evs_np  = [float(x) for x in
                       self._circuit(pnp.array(params, requires_grad=False))]
            decoded = self._decode(evs_np, budget)
            sim_r   = risk_engine.simulate(student, SimulationInput(**decoded))

            if sim_r.projected_score < best_score:
                best_score  = sim_r.projected_score
                best_params = pnp.array(params)

            if verbose and (step + 1) % 10 == 0:
                print(f"[QIO]   Step {step+1:>3}/{n_steps}  "
                      f"projected={sim_r.projected_score:.2f}  "
                      f"best={best_score:.2f}")

            # ── Step 2: quantum gradient step ────────────────────────────
            # Differentiable proxy cost: maximise total weighted intervention
            # strength (correlates with lower risk).  Budget penalty keeps
            # sliders within allowed bounds.
            def circuit_cost(p):
                evs = self._circuit(p)
                # Map each qubit to its slider range
                attend  = (1 + evs[0]) / 2 * 30   # [0, 30]
                acad    = (1 + evs[1]) / 2 * 25   # [0, 25]
                counsel = (1 + evs[2]) / 2 * 10   # [0, 10]
                welfare = (1 + evs[3]) / 2 * 10   # [0, 10] proxy

                # Weighted cost (negative because we minimise in optimizer)
                raw_total = 0.40 * attend + 0.30 * acad + 0.20 * counsel + 0.10 * welfare

                # Budget penalty
                used    = attend + acad + counsel + welfare
                excess  = pnp.maximum(pnp.array(0.0), used - pnp.array(budget))
                penalty = excess * 0.5

                # Negate so minimising this = maximising intervention
                return -raw_total + penalty

            params, _ = opt.step_and_cost(circuit_cost, params)

        # ── Final decode from best params ─────────────────────────────────
        best_evs     = [float(x) for x in
                        self._circuit(pnp.array(best_params, requires_grad=False))]
        best_decoded = self._decode(best_evs, budget)
        final_sim    = risk_engine.simulate(student, SimulationInput(**best_decoded))

        if verbose:
            print(f"[QIO] Done. "
                  f"attend+{best_decoded['attendance_boost']:.1f}  "
                  f"acad+{best_decoded['academic_boost']:.1f}  "
                  f"counsel={best_decoded['counselling_sessions']}  "
                  f"welfare={best_decoded['welfare_support']}")
            print(f"[QIO] Projected: {final_sim.projected_score:.1f}  "
                  f"Reduction: {-final_sim.score_delta:.1f} pts")

        return QuantumOptimisationResult(
            attendance_boost     = round(best_decoded["attendance_boost"], 1),
            academic_boost       = round(best_decoded["academic_boost"], 1),
            counselling_sessions = best_decoded["counselling_sessions"],
            welfare_support      = best_decoded["welfare_support"],
            projected_risk_score = round(final_sim.projected_score, 1),
            risk_reduction       = round(-final_sim.score_delta, 1),
            optimisation_steps   = n_steps,
            method               = "quantum-variational",
        )

    # ── private ──────────────────────────────

    def _build_circuit(self):
        dev = self._dev

        @qml.qnode(dev)
        def circuit(params: pnp.ndarray):
            """
            4-qubit variational circuit.
            Hadamard initialisation → superposition over all states.
            RY + RZ rotations per qubit per layer → expressiveness.
            CNOT ring → entanglement (qubit correlations = intervention correlations).
            """
            for i in range(N_QUBITS_OPTIMIZER):
                qml.Hadamard(wires=i)
            for layer in range(N_LAYERS_OPTIMIZER):
                for q in range(N_QUBITS_OPTIMIZER):
                    qml.RY(params[layer, q, 0], wires=q)
                    qml.RZ(params[layer, q, 1], wires=q)
                # CNOT ring: each qubit entangled with its neighbour
                for q in range(N_QUBITS_OPTIMIZER - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[N_QUBITS_OPTIMIZER - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_OPTIMIZER)]

        return circuit

    @staticmethod
    def _decode(evs: list[float], budget: float) -> dict:
        """
        Decode 4 PauliZ expectation values into intervention slider values.

        Mapping (each expval ∈ (-1, +1) → unit ∈ (0, 1)):
          Qubit 0 → attendance_boost    × 30
          Qubit 1 → academic_boost      × 25
          Qubit 2 → counselling_sessions × 10
          Qubit 3 → welfare_support      × 1
        """
        def unit(e): return (1 + float(e)) / 2

        raw = {
            "attendance_boost"    : unit(evs[0]) * 30.0,
            "academic_boost"      : unit(evs[1]) * 25.0,
            "counselling_sessions": unit(evs[2]) * 10.0,
            "welfare_support"     : unit(evs[3]),
        }

        # Enforce budget by proportional scaling
        total = (raw["attendance_boost"] + raw["academic_boost"]
                 + raw["counselling_sessions"] + raw["welfare_support"] * 10)
        if total > budget and total > 0:
            scale = budget / total
            raw["attendance_boost"]     *= scale
            raw["academic_boost"]       *= scale
            raw["counselling_sessions"] *= scale
            raw["welfare_support"]      *= scale

        # Discretise
        raw["counselling_sessions"] = int(round(raw["counselling_sessions"]))
        raw["welfare_support"]      = (
            0.0 if raw["welfare_support"] < 0.25
            else 0.5 if raw["welfare_support"] < 0.75
            else 1.0
        )
        return raw


# ─────────────────────────────────────────────
#  Integration helper
# ─────────────────────────────────────────────

def get_quantum_risk_score(
    qrc             : QuantumRiskClassifier,
    student_data    : dict,
    student_id      : str,
    rule_based_score: float,
    ml_score        : Optional[float] = None,
    quantum_weight  : float = 0.20,
) -> float:
    """
    Blend the QRC output into the final risk score.

    Three-way blend (when ML is also available):
      50% × rule_based  +  30% × ml_score  +  20% × quantum_score

    Two-way blend (when only rule-based available):
      (1 - quantum_weight) × rule_based  +  quantum_weight × quantum_score

    Returns a blended risk score clamped to [1, 99].
    """
    try:
        q_pred  = qrc.predict(student_id, student_data)
        q_score = q_pred.quantum_risk_score
    except Exception as e:
        print(f"[QRC] Inference failed for {student_id}: {e}. "
              f"Returning rule-based score.")
        return rule_based_score

    if ml_score is not None:
        blended = 0.50 * rule_based_score + 0.30 * ml_score + 0.20 * q_score
    else:
        blended = (1 - quantum_weight) * rule_based_score + quantum_weight * q_score

    return round(max(1.0, min(99.0, blended)), 1)


# ─────────────────────────────────────────────
#  Smoke-test  (python quantum_engine.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Maths.csv"

    print("\n" + "=" * 58)
    print("  EduSight Quantum Engine — Smoke Test")
    print("=" * 58)

    student_data = {
        "absences": 18, "failures": 2,
        "G1": 5,        "G2": 4,
        "Dalc": 3,      "Walc": 4,
    }

    # ── Part A: Quantum Risk Classifier ──────────────────────────
    print("\n[ Part A — Variational Quantum Classifier ]")
    if Path(csv_path).exists():
        qrc = QuantumRiskClassifier()
        qrc.train(csv_path, n_epochs=40, verbose=True)
        qrc.save()
    else:
        print(f"  CSV not found at '{csv_path}', attempting to load saved weights...")
        qrc = QuantumRiskClassifier.load()

    pred = qrc.predict("STU-001", student_data)
    print(f"\n  Student ID          : {pred.student_id}")
    print(f"  Circuit expectation : {pred.circuit_expectation:+.4f}")
    print(f"  Dropout probability : {pred.dropout_probability * 100:.1f}%")
    print(f"  Quantum risk score  : {pred.quantum_risk_score} / 100")
    print(f"  Confidence          : {pred.confidence}")
    print(f"  Architecture        : {pred.n_qubits} qubits × {pred.n_layers} layers")

    blended = get_quantum_risk_score(qrc, student_data, "STU-001",
                                      rule_based_score=82.0, ml_score=75.0)
    print(f"\n  Three-way blend (rule=82, ml=75, q={pred.quantum_risk_score}): {blended}")

    # ── Part B: Quantum Intervention Optimizer ────────────────────
    print("\n" + "=" * 58)
    print("[ Part B — Quantum Intervention Optimizer ]")
    from risk_engine import RiskEngine, StudentProfile, Trend

    risk_engine = RiskEngine(ml_model_path=None)
    student     = StudentProfile(
        student_id="STU-001", name="Muhammad Ali bin Faisal",
        grade="Form 4", attendance_rate=62.0, academic_score=49.0,
        socio_score=55.0, family_support=40.0, trend=Trend.WORSENING,
    )

    qio    = QuantumInterventionOptimizer()
    result = qio.optimise(student, risk_engine, budget=40.0, n_steps=30)

    print(f"\n  ── Quantum-Optimal Intervention Plan ──")
    print(f"  Attendance boost    : +{result.attendance_boost}%")
    print(f"  Academic boost      : +{result.academic_boost} pts")
    print(f"  Counselling sessions: {result.counselling_sessions}")
    print(f"  Welfare support     : {result.welfare_support}")
    print(f"  Projected risk score: {result.projected_risk_score} / 100")
    print(f"  Risk reduction      : −{result.risk_reduction} pts")
    print(f"  Method              : {result.method}")
    print("\n" + "=" * 58 + "\n")
