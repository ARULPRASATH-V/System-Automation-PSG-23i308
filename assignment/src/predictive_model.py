"""
Requirement 2: Predictive Wear Detection Model

Trains and applies a machine learning model on historical sensor telemetry
to detect bearing and joint degradation before catastrophic failure occurs.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from assignment.src.config import HISTORICAL_CSV, MODEL_PATH

FEATURES = [
    "torque_nm",
    "temperature_c",
    "vibration_mms",
    "motor_current_a",
    "speed_rpm",
    "operating_hours",
]

TARGET = "wear_state"

CLASS_RISK_WEIGHTS = {
    "NORMAL": 0.0,
    "EARLY_WEAR": 0.35,
    "ACCELERATED_WEAR": 0.70,
    "CRITICAL_WEAR": 1.0,
}


class PredictiveWearModel:
    def __init__(self, model_file: Path = MODEL_PATH):
        self.model_file = Path(model_file)
        self.model = None
        self.classes = []

    def load_dataset(self, csv_path: Path = HISTORICAL_CSV) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"Historical dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        required = FEATURES + [TARGET]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column in dataset: {col}")

        # Clean and validate
        for col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required)

        # Ensure finite
        finite_mask = np.isfinite(df[FEATURES].to_numpy(dtype=float)).all(axis=1)
        df = df[finite_mask].copy()

        if len(df) < 50:
            raise ValueError(f"Dataset has only {len(df)} rows; requirement is at least 50 labeled rows.")

        return df

    def train(self, csv_path: Path = HISTORICAL_CSV) -> dict:
        df = self.load_dataset(csv_path)
        X = df[FEATURES]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=3,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

        # 5-fold cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        self.model = clf
        self.classes = list(clf.classes_)

        # Ensure directory exists and persist model
        self.model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": FEATURES, "classes": self.classes}, self.model_file)

        return {
            "test_accuracy": round(float(test_acc), 4),
            "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "feature_importances": dict(zip(FEATURES, [round(float(v), 4) for v in clf.feature_importances_])),
            "classification_report": report,
            "sample_count": len(df),
        }

    def load(self) -> bool:
        if not self.model_file.exists():
            return False
        saved = joblib.load(self.model_file)
        self.model = saved["model"]
        self.classes = saved["classes"]
        return True

    def predict_single(self, telemetry: dict) -> dict:
        """
        Takes sensor readings for a joint and predicts wear level and wear risk score.
        If missing features, substitutes reasonable defaults.
        """
        if self.model is None:
            if not self.load():
                # Fallback rule-based wear estimation if model not trained
                return {
                    "wear_state": "UNKNOWN",
                    "wear_risk_score": 0.0,
                    "confidence": 0.0,
                    "probabilities": {},
                    "note": "ML model unavailable; graceful fallback.",
                }

        # Build feature vector
        row = {
            "torque_nm": float(telemetry.get("torque_nm", 50.0)),
            "temperature_c": float(telemetry.get("temperature_c", 45.0)),
            "vibration_mms": float(telemetry.get("vibration_mms", 12.0)),
            "motor_current_a": float(telemetry.get("motor_current_a", telemetry.get("torque_nm", 50.0) / 5.0)),
            "speed_rpm": float(telemetry.get("speed_rpm", 30.0)),
            "operating_hours": float(telemetry.get("operating_hours", 1500.0)),
        }

        X_input = pd.DataFrame([row], columns=FEATURES)
        pred_class = self.model.predict(X_input)[0]
        probs = self.model.predict_proba(X_input)[0]
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.model.classes_, probs)}

        # Risk score calculation [0.0 - 1.0]
        risk_score = 0.0
        for cls, prob in prob_dict.items():
            risk_score += prob * CLASS_RISK_WEIGHTS.get(cls, 0.5)

        confidence = round(float(np.max(probs)), 4)

        return {
            "wear_state": pred_class,
            "wear_risk_score": round(float(risk_score), 4),
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    def evaluate_arm(self, arm_telemetry: dict) -> dict:
        """
        Evaluates predictive wear across all joints of a robotic arm.
        Returns aggregate arm wear state and worst-case joint details.
        """
        joint_predictions = {}
        max_risk = 0.0
        worst_joint = None
        dominant_state = "NORMAL"

        for joint, readings in arm_telemetry.items():
            pred = self.predict_single(readings)
            joint_predictions[joint] = pred
            if pred["wear_risk_score"] > max_risk:
                max_risk = pred["wear_risk_score"]
                worst_joint = joint
                dominant_state = pred["wear_state"]

        return {
            "arm_wear_state": dominant_state,
            "max_wear_risk_score": round(max_risk, 4),
            "worst_joint": worst_joint,
            "joint_predictions": joint_predictions,
        }
