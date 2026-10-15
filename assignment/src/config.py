"""
System Configuration & Engineering Thresholds
Robotic Arm Joint Health Monitoring System (6-Axis Fleet)

Defines:
1. Deterministic safety stop thresholds (independent of ML)
2. Warning thresholds
3. Motor electromechanical constants for physical cross-checks (spoofed-sensor defense)
4. NLP keyword taxonomy for unstructured maintenance logs
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
HISTORICAL_CSV = DATA_DIR / "historical_sensor_readings.csv"
MAINTENANCE_LOG = DATA_DIR / "maintenance_log.txt"
MODEL_PATH = BASE_DIR / "src" / "trained_joint_wear_model.joblib"
REPORTS_DIR = BASE_DIR / "reports"

# 6-Axis joints monitored
JOINTS = ["shoulder", "elbow", "wrist"]

# ----------------------------------------------------------------------
# Requirement 1: Deterministic Safety Stop Thresholds
# Instant hard cutoffs; if ANY reading exceeds these, EMERGENCY_STOP fires immediately.
# ----------------------------------------------------------------------
CRITICAL_THRESHOLDS = {
    "shoulder": {
        "torque_nm": 180.0,       # Exceeds rated structural limit (gearbox teeth fracture risk)
        "temperature_c": 85.0,    # Winding insulation breakdown & thermal expansion lockup
        "vibration_mms": 45.0,    # Severe catastrophic harmonic oscillation / bearing failure
    },
    "elbow": {
        "torque_nm": 120.0,       # Over-moment limit for mid-arm link
        "temperature_c": 80.0,    # Mid-arm thermal ceiling
        "vibration_mms": 38.0,    # Resonant frequency vibration limit
    },
    "wrist": {
        "torque_nm": 45.0,        # Precision planetary gear teeth shear threshold
        "temperature_c": 75.0,    # Harmonic drive flexspline overheating limit
        "vibration_mms": 30.0,    # Tool flange severe vibration limit
    },
}

# Warning / Degradation Thresholds (Used for MONITOR / SCHEDULE_MAINTENANCE guidance)
WARNING_THRESHOLDS = {
    "shoulder": {
        "torque_nm": 130.0,
        "temperature_c": 68.0,
        "vibration_mms": 28.0,
    },
    "elbow": {
        "torque_nm": 90.0,
        "temperature_c": 62.0,
        "vibration_mms": 22.0,
    },
    "wrist": {
        "torque_nm": 32.0,
        "temperature_c": 58.0,
        "vibration_mms": 18.0,
    },
}

# ----------------------------------------------------------------------
# Requirement 4: Electromechanical & Physical Constants (Spoof Defense)
# Motor torque constant Kt (Nm/A) under nominal load
# ----------------------------------------------------------------------
MOTOR_SPECS = {
    "shoulder": {
        "kt_nm_per_amp": 5.0,
        "max_allowable_divergence_pct": 40.0,   # Divergence > 40% triggers spoof suspicion
        "critical_divergence_pct": 70.0,        # Divergence > 70% is definite spoof attack
        "min_sensor_noise_mms": 0.05,          # Flatline check: healthy active sensor must have variance
    },
    "elbow": {
        "kt_nm_per_amp": 5.0,
        "max_allowable_divergence_pct": 40.0,
        "critical_divergence_pct": 70.0,
        "min_sensor_noise_mms": 0.05,
    },
    "wrist": {
        "kt_nm_per_amp": 4.5,
        "max_allowable_divergence_pct": 40.0,
        "critical_divergence_pct": 70.0,
        "min_sensor_noise_mms": 0.05,
    },
}

# ----------------------------------------------------------------------
# Requirement 3: Maintenance Log NLP Keyword Taxonomy
# Free-text scanning for unstructured technician notes
# ----------------------------------------------------------------------
LOG_KEYWORDS = {
    "CRITICAL": [
        "emergency", "severe", "shutdown", "screeching", "smoke", "binding",
        "seized", "catastrophic", "immediate shutdown", "fracture", "burned"
    ],
    "WARNING": [
        "grinding", "discolored", "backlash", "jitter", "warm", "hot",
        "vibration", "play", "seepage", "hum", "loose", "leak", "unusual",
        "gritty", "intermittent", "wear", "noise"
    ],
    "HEALTHY": [
        "smooth", "normal", "secure", "wiped down", "routine", "clean",
        "passed", "green", "cleared", "good", "calibrated"
    ],
    "IRRELEVANT_TOPICS": [
        "coffee", "wifi", "lunch", "lamp", "compressor", "dock", "goggles",
        "gloves", "earplugs", "pizza", "refrigerator", "janitorial", "cisco",
        "bulb", "drain", "fluorescent"
    ],
}

# ----------------------------------------------------------------------
# Requirement 5: Final Decision Enumeration
# ----------------------------------------------------------------------
DECISION_SAFE = "SAFE"
DECISION_MONITOR = "MONITOR"
DECISION_SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
DECISION_EMERGENCY_STOP = "EMERGENCY_STOP"

ALLOWED_DECISIONS = [
    DECISION_SAFE,
    DECISION_MONITOR,
    DECISION_SCHEDULE_MAINTENANCE,
    DECISION_EMERGENCY_STOP,
]
