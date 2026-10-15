"""
Unit tests for Requirement 5: Multi-Signal Decision Engine
Ensures all 4 final decisions are reachable and arbitration priorities hold.
"""

import pytest
from assignment.src.config import (
    DECISION_SAFE,
    DECISION_MONITOR,
    DECISION_SCHEDULE_MAINTENANCE,
    DECISION_EMERGENCY_STOP,
    ALLOWED_DECISIONS,
)
from assignment.src.decision_engine import ArmHealthDecisionEngine
from assignment.main import create_demonstration_fleet


@pytest.fixture(scope="module")
def engine():
    return ArmHealthDecisionEngine()


def test_all_four_decisions_are_reachable(engine):
    fleet = create_demonstration_fleet()
    observed = set()
    for arm_id, data in fleet.items():
        res = engine.evaluate_arm(arm_id, data["telemetry"], data["history"])
        observed.add(res["final_decision"])

    for req_dec in ALLOWED_DECISIONS:
        assert req_dec in observed, f"Decision '{req_dec}' must be reachable in demonstration fleet."


def test_safety_stop_overrides_ml_model(engine):
    # Craft telemetry where ML might predict normal or low risk,
    # but elbow temperature is critically 88 C (> 80 C cutoff)
    telemetry = {
        "shoulder": {"torque_nm": 60.0, "temperature_c": 45.0, "vibration_mms": 10.0, "motor_current_a": 12.0, "speed_rpm": 25.0, "operating_hours": 200},
        "elbow":    {"torque_nm": 50.0, "temperature_c": 88.0, "vibration_mms": 10.0, "motor_current_a": 10.0, "speed_rpm": 30.0, "operating_hours": 200},
        "wrist":    {"torque_nm": 15.0, "temperature_c": 35.0, "vibration_mms": 8.0,  "motor_current_a": 3.0,  "speed_rpm": 50.0, "operating_hours": 200},
    }
    res = engine.evaluate_arm("Arm_Test_Override", telemetry)
    assert res["final_decision"] == DECISION_EMERGENCY_STOP
    assert res["primary_trigger"] == "DETERMINISTIC_SAFETY_VIOLATION"


def test_spoofing_attack_triggers_emergency_stop(engine):
    # Reported torque 15 Nm, but motor current is 29 A (over 140 Nm expected load)
    telemetry = {
        "shoulder": {"torque_nm": 15.0, "temperature_c": 76.0, "vibration_mms": 40.0, "motor_current_a": 29.0, "speed_rpm": 25.0, "operating_hours": 3000},
        "elbow":    {"torque_nm": 50.0, "temperature_c": 50.0, "vibration_mms": 12.0, "motor_current_a": 10.0, "speed_rpm": 30.0, "operating_hours": 3000},
        "wrist":    {"torque_nm": 15.0, "temperature_c": 35.0, "vibration_mms": 8.0,  "motor_current_a": 3.0,  "speed_rpm": 50.0, "operating_hours": 3000},
    }
    res = engine.evaluate_arm("Arm_Test_Spoof", telemetry)
    assert res["final_decision"] == DECISION_EMERGENCY_STOP
    assert "SPOOF" in res["primary_trigger"]


def test_technician_emergency_log_triggers_emergency_stop(engine):
    # Arm_6 has smoke and screeching metallic sound logged
    telemetry = {
        "shoulder": {"torque_nm": 60.0, "temperature_c": 45.0, "vibration_mms": 10.0, "motor_current_a": 12.0, "speed_rpm": 25.0, "operating_hours": 500},
        "elbow":    {"torque_nm": 50.0, "temperature_c": 45.0, "vibration_mms": 10.0, "motor_current_a": 10.0, "speed_rpm": 30.0, "operating_hours": 500},
        "wrist":    {"torque_nm": 15.0, "temperature_c": 35.0, "vibration_mms": 8.0,  "motor_current_a": 3.0,  "speed_rpm": 50.0, "operating_hours": 500},
    }
    res = engine.evaluate_arm("Arm_6", telemetry)
    assert res["final_decision"] == DECISION_EMERGENCY_STOP
    assert res["primary_trigger"] == "TECHNICIAN_LOG_EMERGENCY_DIRECTIVE"
