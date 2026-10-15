"""
Requirement 1: Immediate Safety Stop Guard (Deterministic Rule Engine)

Evaluates raw joint sensor readings directly against hard physical safety thresholds.
Constraint: Must NEVER depend on any Machine Learning model.
Operates instantly with O(1) determinism.
"""

import math
from typing import Dict, Any, List, Tuple
from assignment.src.config import CRITICAL_THRESHOLDS, WARNING_THRESHOLDS, JOINTS

class SafetyViolation:
    def __init__(self, joint: str, metric: str, value: float, threshold: float, severity: str = "CRITICAL"):
        self.joint = joint
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.severity = severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint": self.joint,
            "metric": self.metric,
            "value": round(float(self.value), 2) if isinstance(self.value, (int, float)) else self.value,
            "threshold": self.threshold,
            "severity": self.severity,
            "message": f"{self.joint.upper()} {self.metric} exceeded critical limit ({self.value} > {self.threshold})"
        }


def validate_reading_finite(val: Any) -> bool:
    """Checks if a numerical sensor value is valid and finite."""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


def check_joint_safety(joint: str, readings: Dict[str, Any]) -> List[SafetyViolation]:
    """
    Evaluates a single joint's telemetry against deterministic critical safety thresholds.
    """
    violations = []
    if joint not in CRITICAL_THRESHOLDS:
        return violations

    limits = CRITICAL_THRESHOLDS[joint]

    # 1. Torque check
    torque = readings.get("torque_nm")
    if not validate_reading_finite(torque):
        violations.append(SafetyViolation(joint, "torque_nm", torque, limits["torque_nm"], "CRITICAL_SENSOR_FAULT"))
    elif float(torque) > limits["torque_nm"]:
        violations.append(SafetyViolation(joint, "torque_nm", float(torque), limits["torque_nm"], "CRITICAL"))

    # 2. Temperature check
    temp = readings.get("temperature_c")
    if not validate_reading_finite(temp):
        violations.append(SafetyViolation(joint, "temperature_c", temp, limits["temperature_c"], "CRITICAL_SENSOR_FAULT"))
    elif float(temp) > limits["temperature_c"]:
        violations.append(SafetyViolation(joint, "temperature_c", float(temp), limits["temperature_c"], "CRITICAL"))

    # 3. Vibration check
    vib = readings.get("vibration_mms")
    if not validate_reading_finite(vib):
        violations.append(SafetyViolation(joint, "vibration_mms", vib, limits["vibration_mms"], "CRITICAL_SENSOR_FAULT"))
    elif float(vib) > limits["vibration_mms"]:
        violations.append(SafetyViolation(joint, "vibration_mms", float(vib), limits["vibration_mms"], "CRITICAL"))

    return violations


def check_arm_safety_stop(arm_telemetry: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Checks all joints on a robotic arm.
    Returns:
        (emergency_stop_triggered: bool, list_of_violations: list)
    If ANY single joint crosses a dangerous threshold, emergency_stop_triggered is True.
    """
    all_violations = []

    for joint in JOINTS:
        if joint in arm_telemetry:
            joint_readings = arm_telemetry[joint]
            v_list = check_joint_safety(joint, joint_readings)
            for v in v_list:
                all_violations.append(v.to_dict())

    emergency_stop = len(all_violations) > 0
    return emergency_stop, all_violations


def check_arm_warnings(arm_telemetry: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates softer warning thresholds to detect early mechanical degradation.
    """
    warning_list = []
    for joint in JOINTS:
        if joint not in arm_telemetry:
            continue
        readings = arm_telemetry[joint]
        limits = WARNING_THRESHOLDS.get(joint, {})

        for metric in ["torque_nm", "temperature_c", "vibration_mms"]:
            val = readings.get(metric)
            if validate_reading_finite(val) and metric in limits:
                if float(val) > limits[metric]:
                    warning_list.append({
                        "joint": joint,
                        "metric": metric,
                        "value": float(val),
                        "warn_threshold": limits[metric],
                        "message": f"{joint.upper()} {metric} elevated above warning threshold ({val} > {limits[metric]})"
                    })
    return warning_list
