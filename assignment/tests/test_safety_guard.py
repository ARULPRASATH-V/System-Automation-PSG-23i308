"""
Unit tests for Requirement 1: Deterministic Immediate Safety Stop Guard
"""

import pytest
from assignment.src.safety_guard import check_joint_safety, check_arm_safety_stop, check_arm_warnings


def test_safety_stop_on_shoulder_overtorque():
    # Shoulder limit is 180.0 Nm
    readings = {"torque_nm": 195.0, "temperature_c": 50.0, "vibration_mms": 15.0}
    violations = check_joint_safety("shoulder", readings)
    assert len(violations) == 1
    assert violations[0].metric == "torque_nm"
    assert violations[0].severity == "CRITICAL"


def test_safety_stop_on_elbow_overtemp():
    # Elbow limit is 80.0 C
    readings = {"torque_nm": 60.0, "temperature_c": 84.5, "vibration_mms": 18.0}
    violations = check_joint_safety("elbow", readings)
    assert len(violations) == 1
    assert violations[0].metric == "temperature_c"


def test_safety_stop_on_wrist_excessive_vibration():
    # Wrist limit is 30.0 mm/s
    readings = {"torque_nm": 20.0, "temperature_c": 40.0, "vibration_mms": 34.0}
    violations = check_joint_safety("wrist", readings)
    assert len(violations) == 1
    assert violations[0].metric == "vibration_mms"


def test_arm_safety_stop_triggers_on_any_single_joint():
    arm_telemetry = {
        "shoulder": {"torque_nm": 70.0, "temperature_c": 45.0, "vibration_mms": 12.0},
        "elbow": {"torque_nm": 50.0, "temperature_c": 48.0, "vibration_mms": 10.0},
        "wrist": {"torque_nm": 25.0, "temperature_c": 40.0, "vibration_mms": 36.5},  # Exceeds 30.0 limit
    }
    is_stop, violations = check_arm_safety_stop(arm_telemetry)
    assert is_stop is True
    assert len(violations) == 1
    assert violations[0]["joint"] == "wrist"


def test_nominal_arm_passes_safety_stop():
    arm_telemetry = {
        "shoulder": {"torque_nm": 60.0, "temperature_c": 45.0, "vibration_mms": 12.0},
        "elbow": {"torque_nm": 40.0, "temperature_c": 42.0, "vibration_mms": 10.0},
        "wrist": {"torque_nm": 15.0, "temperature_c": 35.0, "vibration_mms": 8.0},
    }
    is_stop, violations = check_arm_safety_stop(arm_telemetry)
    assert is_stop is False
    assert len(violations) == 0


def test_non_finite_sensor_values_trigger_safety_fault():
    arm_telemetry = {
        "shoulder": {"torque_nm": None, "temperature_c": 45.0, "vibration_mms": 12.0},
        "elbow": {"torque_nm": 40.0, "temperature_c": 42.0, "vibration_mms": 10.0},
        "wrist": {"torque_nm": 15.0, "temperature_c": 35.0, "vibration_mms": 8.0},
    }
    is_stop, violations = check_arm_safety_stop(arm_telemetry)
    assert is_stop is True
    assert violations[0]["severity"] == "CRITICAL_SENSOR_FAULT"


def test_arm_warning_thresholds():
    # Shoulder warning is torque > 130.0 Nm
    arm_telemetry = {
        "shoulder": {"torque_nm": 140.0, "temperature_c": 50.0, "vibration_mms": 15.0},
        "elbow": {"torque_nm": 40.0, "temperature_c": 42.0, "vibration_mms": 10.0},
        "wrist": {"torque_nm": 15.0, "temperature_c": 35.0, "vibration_mms": 8.0},
    }
    is_stop, _ = check_arm_safety_stop(arm_telemetry)
    assert is_stop is False  # Does not trigger emergency stop
    warnings = check_arm_warnings(arm_telemetry)
    assert len(warnings) == 1
    assert warnings[0]["metric"] == "torque_nm"
