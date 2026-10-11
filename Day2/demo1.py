
import ast
import re
import math
from pathlib import Path

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

LOG_FILE = BASE_DIR.parent / "Day1" / "day1_log.txt"
CSV_FILE = BASE_DIR / "machine_sensor_data.csv"
OUTPUT_FILE = BASE_DIR / "log_ml_comparison.csv"

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "rpm",
    "current"
]

ALLOWED_STATES = ["NORMAL", "WARNING", "UNSAFE"]


# --------------------------------------------------
# 2. LOAD AND VALIDATE TRAINING DATA
# --------------------------------------------------

def load_training_data():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {CSV_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    required_columns = FEATURES + ["state"]

    if not all(
        column in df.columns for column in required_columns
    ):
        raise ValueError(
            f"CSV must contain these columns: {required_columns}"
        )

    df = df[required_columns].copy()

    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["state"] = (
        df["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df = df.dropna(subset=required_columns)

    # Remove infinite sensor values.
    finite_rows = df[FEATURES].apply(
        lambda column: column.map(
            lambda value: math.isfinite(float(value))
        )
    ).all(axis=1)

    df = df[finite_rows]

    # Keep only recognized states.
    df = df[
        df["state"].isin(ALLOWED_STATES)
    ].copy()

    if len(df) < 5:
        raise ValueError(
            "At least 5 valid training rows are required."
        )

    if df["state"].nunique() < 2:
        raise ValueError(
            "Training data must contain at least 2 different states."
        )

    print("Training dataset shape:", df.shape)

    print("\nTraining state distribution:")
    print(df["state"].value_counts())

    return df


# --------------------------------------------------
# 3. TRAIN MACHINE-LEARNING MODELS
# --------------------------------------------------

def train_models(df):
    X = df[FEATURES]
    y = df["state"]

    models = {
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),

        "Naive_Bayes": GaussianNB()
    }

    for name, model in models.items():
        model.fit(X, y)

        print(f"{name} trained successfully.")

    return models


# --------------------------------------------------
# 4. CONVERT LOG THINK VALUE INTO MACHINE STATE
# --------------------------------------------------

def determine_logged_state(think):
    if think == "NO_RULE_FIRED":
        return "NORMAL"

    if think in [
        "EXCESSIVE_VIBRATION",
        "HIGH_TEMP",
        "HIGH_PRESSURE"
    ]:
        return "WARNING"

    if think == "CRITICAL_OVERLOAD":
        return "UNSAFE"

    if think == "SENSOR_DROPOUT_DETECTED":
        return "SENSOR_DROPOUT"

    return "UNKNOWN"


# --------------------------------------------------
# 5. READ AND PARSE DAY1 LOG FILE
# --------------------------------------------------

def read_log_file(rpm_fallback):
    if not LOG_FILE.exists():
        raise FileNotFoundError(
            f"Log file not found: {LOG_FILE}"
        )

    records = []

    # Example:
    # [2026-10-10 15:25:25] SENSE: {...}
    # | THINK: NO_RULE_FIRED | ACT: CONTINUE_NORMAL

    pattern = re.compile(
        r"^\[(.*?)\]\s+SENSE:\s+(\{.*?\})"
        r"\s+\|\s+THINK:\s+([A-Z_]+)"
        r"\s+\|\s+ACT:\s+([A-Z_]+)\s*$"
    )

    with open(LOG_FILE, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            match = pattern.search(line.strip())

            if not match:
                continue

            timestamp, sensor_text, think, action = match.groups()

            try:
                sensors = ast.literal_eval(sensor_text)

                if not isinstance(sensors, dict):
                    raise ValueError(
                        "Sensor data must be a dictionary."
                    )

            except (ValueError, SyntaxError):
                print(
                    f"Skipping invalid sensor data "
                    f"on line {line_number}"
                )
                continue

            record = {
                "log_line": line_number,
                "timestamp": timestamp,
                "temperature": sensors.get("temperature"),
                "pressure": sensors.get("pressure"),
                "vibration": sensors.get("vibration"),
                "current": sensors.get("current"),
                "logged_think": think,
                "logged_action": action
            }

            # Map the logged rule to a machine state.
            logged_state = determine_logged_state(think)

            record["logged_state"] = logged_state

            # Handle missing or invalid sensor readings.
            sensor_values = [
                record["temperature"],
                record["pressure"],
                record["vibration"],
                record["current"]
            ]

            sensor_dropout = any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in sensor_values
            )

            if (
                sensor_dropout
                or logged_state == "SENSOR_DROPOUT"
            ):
                record["logged_state"] = "SENSOR_DROPOUT"
                record["rpm_used"] = None
                record["rpm_assumed"] = False

                for name in [
                    "Decision_Tree",
                    "Random_Forest",
                    "Naive_Bayes"
                ]:
                    record[name] = "NOT_PREDICTED"

                record["ml_majority"] = "NOT_PREDICTED"
                record["comparison"] = "SENSOR_DROPOUT"

                records.append(record)
                continue

            # RPM is absent from the log.
            record["rpm_used"] = rpm_fallback
            record["rpm_assumed"] = True

            records.append(record)

    if not records:
        raise ValueError(
            "No valid log records found. Check the log format."
        )

    return records


# --------------------------------------------------
# 6. PREDICT AND COMPARE MODEL RESULTS
# --------------------------------------------------

def compare_predictions():
    df = load_training_data()

    models = train_models(df)

    # The log has no RPM field, so use the dataset median.
    rpm_fallback = float(df["rpm"].median())

    records = read_log_file(rpm_fallback)

    for record in records:

        # Do not predict if the sensor data is incomplete.
        if record["logged_state"] == "SENSOR_DROPOUT":
            continue

        sample = pd.DataFrame(
            [{
                "temperature": record["temperature"],
                "pressure": record["pressure"],
                "vibration": record["vibration"],
                "rpm": record["rpm_used"],
                "current": record["current"]
            }],
            columns=FEATURES
        )

        predictions = []

        # Get a prediction from each model.
        for name, model in models.items():
            prediction = model.predict(sample)[0]

            record[name] = prediction
            predictions.append(prediction)

        # Majority voting.
        counts = pd.Series(predictions).value_counts()

        if (
            len(counts) > 1
            and counts.iloc[0] == counts.iloc[1]
        ):
            majority_state = "DISAGREEMENT"
        else:
            majority_state = counts.index[0]

        record["ml_majority"] = majority_state

        # Compare the logged state with the ML prediction.
        if record["logged_state"] == "UNKNOWN":
            record["comparison"] = "UNKNOWN_LOG_STATE"

        elif majority_state == "DISAGREEMENT":
            record["comparison"] = "MODEL_DISAGREEMENT"

        elif record["logged_state"] == majority_state:
            record["comparison"] = "MATCH"

        else:
            record["comparison"] = "DIFFERENT"

    # Save all comparison records.
    result = pd.DataFrame(records)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n========== LOG COMPARISON ==========")

    print("Log entries processed:", len(result))

    print("RPM used when missing:", rpm_fallback)

    print("RPM was estimated because it is absent from the log.")

    print("\n========== COMPARISON SUMMARY ==========")

    print(
        result["comparison"]
        .value_counts()
        .to_string()
    )

    columns = [
        "log_line",
        "logged_think",
        "logged_state",
        "rpm_used",
        "rpm_assumed",
        "Decision_Tree",
        "Random_Forest",
        "Naive_Bayes",
        "ml_majority",
        "comparison"
    ]

    print("\n========== DETAILED RESULTS ==========")

    print(
        result[columns].to_string(index=False)
    )

    print("\nComparison saved to:", OUTPUT_FILE)


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    try:
        compare_predictions()

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pd.errors.ParserError
    ) as error:
        print("Error:", error)
