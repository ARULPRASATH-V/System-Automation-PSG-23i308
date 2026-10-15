"""
Unit tests for Requirement 4: Spoofed-Sensor Defense Engine
"""

import pytest
from assignment.src.spoof_defense import SpoofDefenseEngine


def test_detects_electromechanical_current_spoof():
    engine = SpoofDefenseEngine()
    # Shoulder Kt is 5.0 Nm/A. Motor current = 28.0 A implies ~140 Nm.
    # Attacker reports fake low torque of 18.0 Nm to mask overload.
    res = engine.check_electromechanical_consistency("shoulder", torque_nm=18.0, current_a=28.0)
    assert res is not None
    assert res.is_spoofed is True
    assert res.spoof_type == "ELECTROMECHANICAL_CURRENT_MISMATCH"
    assert res.confidence >= 0.70
    assert res.estimated_actual_torque >= 130.0


def test_detects_thermodynamic_divergence_spoof():
    engine = SpoofDefenseEngine()
    # High temp (82 C) and severe vibration (45 mm/s) while torque claimed to be 12 Nm
    res = engine.check_thermodynamic_divergence("shoulder", torque_nm=12.0, temp_c=82.0, vib_mms=45.0)
    assert res is not None
    assert res.is_spoofed is True
    assert res.spoof_type == "THERMAL_VIBRATION_ENERGY_DIVERGENCE"
    assert res.confidence >= 0.85


def test_detects_sensor_flatline():
    engine = SpoofDefenseEngine()
    # 5 identical readings without ADC jitter/variance
    flat_history = [
        {"torque_nm": 45.000, "vibration_mms": 12.000},
        {"torque_nm": 45.000, "vibration_mms": 12.000},
        {"torque_nm": 45.000, "vibration_mms": 12.000},
        {"torque_nm": 45.000, "vibration_mms": 12.000},
    ]
    res = engine.check_sensor_flatline("elbow", flat_history)
    assert res is not None
    assert res.is_spoofed is True
    assert res.spoof_type == "SENSOR_FLATLINE_INJECTION"


def test_normal_telemetry_passes_spoof_check():
    engine = SpoofDefenseEngine()
    # Shoulder: torque = 75 Nm, current = 15 A (75 / 5.0 = 15 A, 0% divergence)
    res = engine.check_electromechanical_consistency("shoulder", torque_nm=75.0, current_a=15.0)
    assert res is None  # No spoof violation
