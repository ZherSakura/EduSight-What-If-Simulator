"""
EduSight — Machine Learning Engine
====================================
Trains a dropout-risk classifier and grade predictor on the UCI Student Performance dataset (Maths.csv / Portuguese.csv).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

# Grade threshold: students scoring below this are "at risk"
AT_RISK_THRESHOLD = 10        # out of 20 (Malaysian passing grade equivalent)
DROPOUT_GRADE    = 0          # G3 == 0 → complete dropout / did not sit exam

# Default model save path
DEFAULT_MODEL_PATH = Path("edusight_ml_model.joblib")

# Categorical columns needing encoding
BINARY_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

NOMINAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]

NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
    "G1", "G2",   # G1/G2 are strong predictors; G3 is the target
]

ALL_FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + NOMINAL_COLS


# ─────────────────────────────────────────────
#  Output schemas
# ─────────────────────────────────────────────

@dataclass
class MLRiskPrediction:
    """
    Single-student ML inference result.
    Consumed by risk_engine.py to enrich/override the rule-based score.
    """
    student_id          : str
    dropout_probability : float   # P(G3 == 0)
    atrisk_probability  : float   # P(G3 < 10)
    predicted_g3        : float   # regression estimate of final grade
    ml_risk_score       : float   # 0–100 composite (fed into RiskEngine)
    top_risk_features   : list[tuple[str, float]]  # (feature_name, importance)
    confidence          : str     # "high" / "medium" / "low"


@dataclass
class ModelMetrics:
    """Training evaluation results."""
    accuracy          : float
    roc_auc           : float
    cv_roc_auc_mean   : float
    cv_roc_auc_std    : float
    classification_rep: str
    confusion_matrix  : list[list[int]]
    feature_importances: dict[str, float]


# ─────────────────────────────────────────────
#  Data loader & feature engineer
# ─────────────────────────────────────────────

class StudentDataLoader:
    """
    Loads one or more CSV/xlsx student performance files,
    engineers features, and returns a model-ready DataFrame.

    Handles both the 'Maths.csv' and 'Portuguese.csv' datasets.
    If both are provided they are concatenated with a 'course' column added.
    """

    def load(self, *file_paths: str | Path) -> pd.DataFrame:
        frames = []
        for path in file_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path}")
            df = self._read_file(path)
            df["course"] = path.stem   # 'Maths' or 'Portuguese'
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined = self._engineer_features(combined)
        return combined

    # ── private ──────────────────────────────

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        """Auto-detect csv vs xlsx (Maths.csv is actually an xlsx)."""
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            pass
        try:
            return pd.read_csv(path, sep=",")
        except Exception:
            pass
        # Fallback: Excel (the uploaded file is actually xlsx)
        return pd.read_excel(path)

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features that improve predictive power.
        All new columns are prefixed with 'feat_'.
        """
        # Grade momentum: improvement from G1 to G2
        if "G1" in df.columns and "G2" in df.columns:
            df["feat_grade_momentum"] = df["G2"] - df["G1"]
            df["feat_avg_prior_grade"] = (df["G1"] + df["G2"]) / 2

        # Alcohol risk index
        if "Dalc" in df.columns and "Walc" in df.columns:
            df["feat_alcohol_risk"] = (df["Dalc"] * 2 + df["Walc"]) / 3

        # Parental education average
        if "Medu" in df.columns and "Fedu" in df.columns:
            df["feat_parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

        # Isolation index (romantic + goout inverse + freetime inverse)
        if all(c in df.columns for c in ["romantic", "goout", "freetime"]):
            romantic_num = (df["romantic"] == "yes").astype(int)
            df["feat_isolation_index"] = (
                romantic_num + (5 - df["goout"]) + (5 - df["freetime"])
            ) / 3

        # Absence severity bucket
        if "absences" in df.columns:
            df["feat_absence_bucket"] = pd.cut(
                df["absences"],
                bins=[-1, 0, 5, 15, 100],
                labels=[0, 1, 2, 3],
            ).astype(int)

        # Study-to-failure ratio
        if "studytime" in df.columns and "failures" in df.columns:
            df["feat_study_fail_ratio"] = df["studytime"] / (df["failures"] + 1)

        # Binary dropout and at-risk labels
        df["label_dropout"] = (df["G3"] == DROPOUT_GRADE).astype(int)
        df["label_atrisk"]  = (df["G3"] < AT_RISK_THRESHOLD).astype(int)

        return df


# ─────────────────────────────────────────────
#  Encoder helper (fits on train, transforms both)
# ─────────────────────────────────────────────

class FeatureEncoder:
    """Label-encodes categorical columns; stores mappings for inference."""

    def __init__(self):
        self._encoders: dict[str, LabelEncoder] = {}
        self.feature_names_: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Binary yes/no columns → 0/1
        for col in BINARY_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le

        # Nominal columns → integer codes
        for col in NOMINAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le

        # 'course' column if present
        if "course" in df.columns:
            le = LabelEncoder()
            df["course"] = le.fit_transform(df["course"].astype(str))
            self._encoders["course"] = le

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, le in self._encoders.items():
            if col in df.columns:
                # Handle unseen labels gracefully
                classes = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in classes else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df


# ─────────────────────────────────────────────
#  Main ML Engine
# ─────────────────────────────────────────────

class MLEngine:
    """
    Trains, evaluates, and serves ML predictions for EduSight.

    Three models are trained internally:
      1. Random Forest         → dropout classifier  (primary)
      2. Gradient Boosting     → at-risk classifier  (secondary)
      3. Logistic Regression   → calibrated probability (ensemble blend)

    The final ml_risk_score is a weighted blend:
      60% × P(at_risk)_GB  +  40% × P(dropout)_RF  → scaled to 0–100

    Usage
    -----
    engine = MLEngine()
    engine.train("Maths.csv")                        # or both CSVs
    engine.train("Maths.csv", "Portuguese.csv")
    metrics = engine.evaluate()
    engine.save()                                    # persist model

    # Later / in production:
    engine = MLEngine.load()
    prediction = engine.predict_risk("STU-001", student_row_dict)
    """

    def __init__(self):
        self._loader   = StudentDataLoader()
        self._encoder  = FeatureEncoder()
        self._scaler   = StandardScaler()

        # Models
        self._rf_dropout : Optional[RandomForestClassifier]      = None
        self._gb_atrisk  : Optional[GradientBoostingClassifier]  = None
        self._lr_blend   : Optional[LogisticRegression]          = None

        # Metadata
        self._feature_cols  : list[str] = []
        self._is_trained     : bool     = False
        self._train_df       : Optional[pd.DataFrame] = None

    # ── public API ──────────────────────────

    def train(self, *csv_paths: str | Path) -> "MLEngine":
        """
        Load data, encode features, train all models.

        Parameters
        ----------
        *csv_paths : one or more paths to Maths.csv / Portuguese.csv
        """
        print("[EduSight ML] Loading data...")
        df = self._loader.load(*csv_paths)
        print(f"[EduSight ML] Loaded {len(df)} student records, "
              f"{df['label_dropout'].sum()} dropout cases, "
              f"{df['label_atrisk'].sum()} at-risk cases.")

        # Encode
        df_enc = self._encoder.fit_transform(df)

        # Build feature matrix
        self._feature_cols = self._resolve_features(df_enc)
        X = df_enc[self._feature_cols].fillna(0).values
        y_dropout = df_enc["label_dropout"].values
        y_atrisk  = df_enc["label_atrisk"].values

        # Scale
        X_scaled = self._scaler.fit_transform(X)

        # Train/test split (stratify on dropout — minority class)
        X_tr, X_te, yd_tr, yd_te, ya_tr, ya_te = train_test_split(
            X_scaled, y_dropout, y_atrisk,
            test_size=0.2, random_state=42, stratify=y_dropout,
        )

        # ── Model 1: Random Forest → dropout ──
        print("[EduSight ML] Training Random Forest (dropout classifier)...")
        dropout_weights = compute_class_weight(
            "balanced", classes=np.unique(yd_tr), y=yd_tr
        )
        self._rf_dropout = RandomForestClassifier(
            n_estimators    = 300,
            max_depth       = 8,
            min_samples_leaf= 4,
            class_weight    = {0: dropout_weights[0], 1: dropout_weights[1]},
            random_state    = 42,
            n_jobs          = -1,
        )
        self._rf_dropout.fit(X_tr, yd_tr)

        # ── Model 2: Gradient Boosting → at-risk ──
        print("[EduSight ML] Training Gradient Boosting (at-risk classifier)...")
        self._gb_atrisk = GradientBoostingClassifier(
            n_estimators   = 200,
            learning_rate  = 0.05,
            max_depth      = 4,
            subsample      = 0.8,
            random_state   = 42,
        )
        self._gb_atrisk.fit(X_tr, ya_tr)

        # ── Model 3: Logistic Regression → blend ──
        print("[EduSight ML] Training Logistic Regression (blend layer)...")
        p_dropout_tr = self._rf_dropout.predict_proba(X_tr)[:, 1]
        p_atrisk_tr  = self._gb_atrisk.predict_proba(X_tr)[:, 1]
        blend_tr     = np.column_stack([p_dropout_tr, p_atrisk_tr])
        self._lr_blend = LogisticRegression(random_state=42)
        self._lr_blend.fit(blend_tr, yd_tr)

        # Store test set for evaluate()
        self._X_te      = X_te
        self._yd_te     = yd_te
        self._ya_te     = ya_te
        self._train_df  = df_enc
        self._is_trained = True
        print("[EduSight ML] Training complete.")
        return self

    def evaluate(self) -> ModelMetrics:
        """Run evaluation on held-out test set and return metrics."""
        self._assert_trained()

        yd_pred = self._rf_dropout.predict(self._X_te)
        yd_prob = self._rf_dropout.predict_proba(self._X_te)[:, 1]

        # Cross-validated ROC-AUC (5-fold)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_all = self._scaler.transform(
            self._train_df[self._feature_cols].fillna(0).values
        )
        cv_scores = cross_val_score(
            self._rf_dropout,
            X_all,
            self._train_df["label_dropout"].values,
            cv=cv, scoring="roc_auc",
        )

        acc    = (yd_pred == self._yd_te).mean()
        roc    = roc_auc_score(self._yd_te, yd_prob)
        cm     = confusion_matrix(self._yd_te, yd_pred).tolist()
        report = classification_report(self._yd_te, yd_pred,
                                        target_names=["Enrolled", "Dropout"])

        importances = dict(zip(
            self._feature_cols,
            self._rf_dropout.feature_importances_.round(4).tolist(),
        ))
        importances = dict(sorted(importances.items(),
                                   key=lambda x: x[1], reverse=True))

        metrics = ModelMetrics(
            accuracy          = round(acc, 4),
            roc_auc           = round(roc, 4),
            cv_roc_auc_mean   = round(cv_scores.mean(), 4),
            cv_roc_auc_std    = round(cv_scores.std(), 4),
            classification_rep= report,
            confusion_matrix  = cm,
            feature_importances= importances,
        )

        print("\n=== EduSight ML Model Evaluation ===")
        print(f"Accuracy      : {metrics.accuracy:.4f}")
        print(f"ROC-AUC       : {metrics.roc_auc:.4f}")
        print(f"CV ROC-AUC    : {metrics.cv_roc_auc_mean:.4f} ± {metrics.cv_roc_auc_std:.4f}")
        print(f"\nClassification Report:\n{metrics.classification_rep}")
        print(f"Confusion Matrix: {metrics.confusion_matrix}")
        print("\nTop 10 Feature Importances:")
        for feat, imp in list(importances.items())[:10]:
            bar = "█" * int(imp * 100)
            print(f"  {feat:<30} {imp:.4f}  {bar}")
        return metrics

    def predict_risk(
        self,
        student_id  : str,
        student_data: dict,
    ) -> MLRiskPrediction:
        """
        Real-time risk prediction for a single student.

        Parameters
        ----------
        student_id   : unique ID string (e.g. "STU-001")
        student_data : dict mapping column names → values.
                       Must include all feature columns.
                       G1 and G2 should be present for best accuracy.
                       G3 should be omitted (it is the target).

        Returns
        -------
        MLRiskPrediction with probabilities and top risk features.

        Integration note for Person 3
        ------------------------------
        Call this from POST /simulate after receiving the student
        profile from GET /students/:id. Map the API response fields
        to the column names expected here (see NUMERIC_COLS etc.)
        """
        self._assert_trained()

        row = pd.DataFrame([student_data])
        row = self._loader._engineer_features(row)
        row = self._encoder.transform(row)
        feats = [c for c in self._feature_cols if c in row.columns]
        X = row[feats].fillna(0).reindex(columns=self._feature_cols, fill_value=0).values
        X_scaled = self._scaler.transform(X)

        # Predictions
        p_dropout = float(self._rf_dropout.predict_proba(X_scaled)[0, 1])
        p_atrisk  = float(self._gb_atrisk.predict_proba(X_scaled)[0, 1])

        # Blend layer
        blend     = np.array([[p_dropout, p_atrisk]])
        p_final   = float(self._lr_blend.predict_proba(blend)[0, 1])

        # Composite ML risk score (0–100)
        ml_risk   = round((0.40 * p_dropout + 0.60 * p_atrisk) * 100, 1)

        # Grade estimate: weighted avg of G1/G2 with failure penalty
        g1 = student_data.get("G1", 10)
        g2 = student_data.get("G2", 10)
        failures = student_data.get("failures", 0)
        pred_g3 = round(max(0.0, (0.35 * g1 + 0.65 * g2) - failures * 1.5), 1)

        # Top risk features for this student (SHAP-lite: importance × |deviation|)
        top_feats = self._top_features(X_scaled[0], n=5)

        # Confidence based on G1/G2 availability
        confidence = "high" if ("G1" in student_data and "G2" in student_data) else "medium"

        return MLRiskPrediction(
            student_id          = student_id,
            dropout_probability = round(p_dropout, 3),
            atrisk_probability  = round(p_atrisk, 3),
            predicted_g3        = pred_g3,
            ml_risk_score       = ml_risk,
            top_risk_features   = top_feats,
            confidence          = confidence,
        )

    def predict_batch(
        self,
        students: list[dict],
        id_col  : str = "student_id",
    ) -> list[MLRiskPrediction]:
        """
        Bulk prediction — e.g. score an entire class or school.
        students is a list of dicts, each with the same keys as predict_risk.
        """
        self._assert_trained()
        results = []
        for s in students:
            sid = s.get(id_col, "UNKNOWN")
            try:
                pred = self.predict_risk(sid, {k: v for k, v in s.items() if k != id_col})
                results.append(pred)
            except Exception as e:
                print(f"[MLEngine] Warning: could not score student {sid}: {e}")
        return results

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Persist the trained model artefacts to disk."""
        self._assert_trained()
        payload = {
            "rf_dropout"   : self._rf_dropout,
            "gb_atrisk"    : self._gb_atrisk,
            "lr_blend"     : self._lr_blend,
            "encoder"      : self._encoder,
            "scaler"       : self._scaler,
            "feature_cols" : self._feature_cols,
        }
        joblib.dump(payload, path)
        print(f"[EduSight ML] Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "MLEngine":
        """Load a previously saved model from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Run MLEngine().train('Maths.csv').save() first."
            )
        payload = joblib.load(path)
        engine = cls()
        engine._rf_dropout    = payload["rf_dropout"]
        engine._gb_atrisk     = payload["gb_atrisk"]
        engine._lr_blend      = payload["lr_blend"]
        engine._encoder       = payload["encoder"]
        engine._scaler        = payload["scaler"]
        engine._feature_cols  = payload["feature_cols"]
        engine._is_trained    = True
        print(f"[EduSight ML] Model loaded ← {path}")
        return engine

    # ── private helpers ──────────────────────

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        """Return all usable feature columns present in the DataFrame."""
        engineered = [c for c in df.columns if c.startswith("feat_")]
        base = [c for c in NUMERIC_COLS + BINARY_COLS + NOMINAL_COLS
                if c in df.columns]
        if "course" in df.columns:
            base.append("course")
        return base + engineered

    def _top_features(self, x_row: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
        """
        Approximate per-sample feature importance.
        Score = global_importance × |z-score deviation from mean|.
        Returns top-n (feature_name, contribution_score) pairs.
        """
        global_imp = self._rf_dropout.feature_importances_
        contributions = global_imp * np.abs(x_row)
        top_idx = contributions.argsort()[::-1][:n]
        return [
            (self._feature_cols[i], round(float(contributions[i]), 4))
            for i in top_idx
        ]

    def _assert_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Model is not trained. Call .train('Maths.csv') first, "
                "or load a saved model with MLEngine.load()."
            )


# ─────────────────────────────────────────────
#  Integration bridge for risk_engine.py
# ─────────────────────────────────────────────

def ml_risk_to_student_profile_override(
    ml_pred         : MLRiskPrediction,
    rule_based_score: float,
    blend_weight    : float = 0.45,
) -> float:
    """
    Blend ML risk score with the rule-based score from risk_engine.py.

    The blended score replaces 'total_score' in RiskScore when ML
    predictions are available. This gives the best of both worlds:
      - ML captures patterns in historical data (non-linear interactions)
      - Rule-based engine incorporates real-time slider adjustments

    Formula
    -------
      final = (1 - w) × rule_based  +  w × ml_risk
      default w = 0.45

    Parameters
    ----------
    ml_pred          : output of MLEngine.predict_risk()
    rule_based_score : RiskScore.total_score from RiskEngine.score()
    blend_weight     : how much weight to give the ML model (0–1)

    Returns
    -------
    Blended risk score (0–100), rounded to 1 dp.
    """
    blended = (1 - blend_weight) * rule_based_score + blend_weight * ml_pred.ml_risk_score
    return round(max(1.0, min(99.0, blended)), 1)


# ─────────────────────────────────────────────
#  Smoke-test  (python ml_engine.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Default path — adjust if your CSV is elsewhere
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Maths.csv"

    if not Path(csv_path).exists():
        # Try the uploads directory used in the Claude sandbox
        csv_path = "/tmp/Maths.xlsx"

    print(f"\n{'='*55}")
    print("  EduSight ML Engine — Training & Evaluation")
    print(f"{'='*55}\n")

    engine = MLEngine()
    engine.train(csv_path)
    metrics = engine.evaluate()
    engine.save("edusight_ml_model.joblib")

    print(f"\n{'='*55}")
    print("  Single-student inference demo")
    print(f"{'='*55}\n")

    # Simulate a high-risk student (mirrors Muhammad Ali bin Faisal)
    student_data = {
        "school"    : "GP",
        "sex"       : "M",
        "age"       : 17,
        "address"   : "U",
        "famsize"   : "GT3",
        "Pstatus"   : "T",
        "Medu"      : 2,
        "Fedu"      : 1,
        "Mjob"      : "other",
        "Fjob"      : "other",
        "reason"    : "home",
        "guardian"  : "mother",
        "traveltime": 2,
        "studytime" : 1,
        "failures"  : 2,
        "schoolsup" : "no",
        "famsup"    : "no",
        "paid"      : "no",
        "activities": "no",
        "nursery"   : "yes",
        "higher"    : "no",
        "internet"  : "no",
        "romantic"  : "yes",
        "famrel"    : 2,
        "freetime"  : 4,
        "goout"     : 4,
        "Dalc"      : 3,
        "Walc"      : 4,
        "health"    : 2,
        "absences"  : 18,
        "G1"        : 5,
        "G2"        : 4,
    }

    pred = engine.predict_risk("STU-001", student_data)
    print(f"Student ID         : {pred.student_id}")
    print(f"Dropout Probability: {pred.dropout_probability * 100:.1f}%")
    print(f"At-Risk Probability: {pred.atrisk_probability * 100:.1f}%")
    print(f"Predicted G3       : {pred.predicted_g3} / 20")
    print(f"ML Risk Score      : {pred.ml_risk_score} / 100")
    print(f"Confidence         : {pred.confidence}")
    print(f"Top Risk Features  :")
    for feat, score in pred.top_risk_features:
        print(f"   {feat:<30} {score:.4f}")

    # Show blend with rule-based score
    blended = ml_risk_to_student_profile_override(pred, rule_based_score=82.0)
    print(f"\nBlended Score (ML + Rule-Based 82.0): {blended}")

    print(f"\n{'='*55}")
    print("  Done. Model saved as edusight_ml_model.joblib")
    print(f"{'='*55}\n")
