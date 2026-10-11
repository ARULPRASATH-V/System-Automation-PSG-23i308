import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("data/day2_sample.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

required_columns = [
    "temperature", "pressure", "vibration",
    "current", "rpm", "state"
]

if list(df.columns) != required_columns:
    raise ValueError(f"Incorrect columns. Expected: {required_columns}")

if df.isnull().any().any():
    raise ValueError("Dataset contains missing values.")

if not np.isfinite(
    df[required_columns[:-1]].to_numpy(dtype=float)
).all():
    raise ValueError("Dataset contains invalid numeric values.")

if not df["state"].isin([
    "NORMAL",
    "MOTOR_OVERLOAD",
    "OVERHEATING",
    "LOW_PRESSURE",
    "HIGH_VIBRATION"
]).all():
    raise ValueError("Dataset contains unknown state labels.")

X = df[
    ["temperature", "pressure", "vibration", "current", "rpm"]
]

y = df["state"]

print("\nState Distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n========== MODEL ACCURACY ==========")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_test_pred, zero_division=0))

print("\n========== CONFUSION MATRIX ==========")
print("Class Order:", model.classes_)
print(confusion_matrix(y_test, y_test_pred))

print("\n========== FEATURE IMPORTANCE ==========")
for feature, importance in sorted(
    zip(X.columns, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{feature}: {importance:.4f}")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    DecisionTreeClassifier(random_state=42),
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n========== CROSS VALIDATION ==========")
print("Fold Accuracies:", cv_scores * 100)
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")

joblib.dump(model, "decision_tree_model.joblib")
print("\nModel saved as decision_tree_model.joblib")

sample = pd.DataFrame([{
    "temperature": 70.0,
    "pressure": 5.0,
    "vibration": 2.0,
    "current": 12.0,
    "rpm": 1800
}])

prediction = model.predict(sample)

print("\n========== NEW SENSOR PREDICTION ==========")
print("Predicted Machine State:", prediction[0])

