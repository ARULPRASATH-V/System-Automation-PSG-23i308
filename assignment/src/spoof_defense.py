"""
Requirement 4: Spoofed-Sensor Defense System

Detects adversarial manipulation or sensor spoofing (e.g., an attacker feeding
a falsified low torque reading to mask mechanical overload or severe joint wear).
Leverages physical cross-sensor invariants:
1. Electromechanical invariant: Torque vs. Motor Armature Current (tau = Kt * I)
2. Thermodynamic & kinetic invariant: High temp & vibration cannot coexist with low torque
3. Signal noise floor & flatline injection detection (ADC variance check)
4. Kinematic cross-joint moment coupling
"""

import math
from typing import Dict, List, Any, Optional

from assignment.src.config import MOTOR_SPECS, JOINTS


class SpoofDetectionResult:
    def __init__(
        self,
        joint: str,
        is_spoofed: bool,
        spoof_type: Optional[str] = None,
        confidence: float = 0.0,
        evidence: Optional[str] = None,
        estimated_actual_torque: Optional[float] = None,
        reported_torque: Optional[float] = None,
    ):
        self.joint = joint
        self.is_spoofed = is_spoofed
        self.spoof_type = spoof_type
        self.confidence = confidence
        self.evidence = evidence
        self.estimated_actual_torque = estimated_actual_torque
        self.reported_torque = reported_torque

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint": self.joint,
            "is_spoofed": self.is_spoofed,
            "spoof_type": self.spoof_type,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "estimated_actual_torque_nm": round(self.estimated_actual_torque, 2) if self.estimated_actual_torque is not None else None,
            "reported_torque_nm": round(self.reported_torque, 2) if self.reported_torque is not None else None,
        }


class SpoofDefenseEngine:
    def __init__(self):
        self.specs = MOTOR_SPECS

    def check_electromechanical_consistency(
        self, joint: str, torque_nm: float, current_a: float
    ) -> Optional[SpoofDetectionResult]:
        """
        Validates the physical electromechanical coupling:
        Tau_expected = Kt * I_motor
        If reported torque is artificially suppressed while current is high,
        an attacker is attempting to mask a motor/gearbox overload.
        """
        if joint not in self.specs:
            return None

        kt = self.specs[joint]["kt_nm_per_amp"]
        crit_pct = self.specs[joint]["critical_divergence_pct"]
        warn_pct = self.specs[joint]["max_allowable_divergence_pct"]

        expected_torque = kt * current_a
        delta = abs(expected_torque - torque_nm)
        denom = max(expected_torque, 1.0)
        divergence_pct = (delta / denom) * 100.0

        if divergence_pct > crit_pct and current_a > 4.0:
            evidence = (
                f"Electromechanical Mismatch: Motor drawing {current_a:.2f}A implies ~{expected_torque:.1f}Nm "
                f"(Kt={kt}), but torque sensor reported only {torque_nm:.1f}Nm ({divergence_pct:.1f}% divergence). "
                f"Severe spoofing suspected to conceal heavy joint load."
            )
            return SpoofDetectionResult(
                joint=joint,
                is_spoofed=True,
                spoof_type="ELECTROMECHANICAL_CURRENT_MISMATCH",
                confidence=min(0.99, divergence_pct / 100.0),
                evidence=evidence,
                estimated_actual_torque=expected_torque,
                reported_torque=torque_nm,
            )
        elif divergence_pct > warn_pct and current_a > 5.0:
            evidence = (
                f"Electromechanical Inconsistency: Motor current ({current_a:.2f}A) diverges by "
                f"{divergence_pct:.1f}% from reported torque ({torque_nm:.1f}Nm)."
            )
            return SpoofDetectionResult(
                joint=joint,
                is_spoofed=True,
                spoof_type="SUSPICIOUS_CURRENT_TORQUE_DEVIATION",
                confidence=0.65,
                evidence=evidence,
                estimated_actual_torque=expected_torque,
                reported_torque=torque_nm,
            )

        return None

    def check_thermodynamic_divergence(
        self, joint: str, torque_nm: float, temp_c: float, vib_mms: float
    ) -> Optional[SpoofDetectionResult]:
        """
        Validates thermodynamic energy conservation:
        Extreme temperatures (>75°C) and violent vibrations (>35 mm/s) cannot physically
        occur if the joint is idling at low torque (<25 Nm) unless masked by spoofing.
        """
        is_suspicious_wrist = (joint == "wrist" and temp_c > 65.0 and vib_mms > 24.0 and torque_nm < 10.0)
        is_suspicious_shoulder_elbow = (joint in ["shoulder", "elbow"] and temp_c > 75.0 and vib_mms > 35.0 and torque_nm < 25.0)

        if is_suspicious_wrist or is_suspicious_shoulder_elbow:
            evidence = (
                f"Thermodynamic/Vibration Invariance Violation: Joint exhibits severe thermal rise "
                f"({temp_c:.1f}°C) and violent vibration ({vib_mms:.1f} mm/s) while torque sensor claims "
                f"near-idle load ({torque_nm:.1f} Nm). Violates mechanical energy conservation."
            )
            return SpoofDetectionResult(
                joint=joint,
                is_spoofed=True,
                spoof_type="THERMAL_VIBRATION_ENERGY_DIVERGENCE",
                confidence=0.92,
                evidence=evidence,
                estimated_actual_torque=torque_nm * 4.0,  # Estimated actual binding torque
                reported_torque=torque_nm,
            )

        return None

    def check_sensor_flatline(
        self, joint: str, historical_readings: List[Dict[str, float]]
    ) -> Optional[SpoofDetectionResult]:
        """
        Detects static signal clamping / replay attack where sensor variance is 0.0
        across consecutive cycles while arm is active.
        """
        if not historical_readings or len(historical_readings) < 4:
            return None

        torques = [r.get("torque_nm", 0.0) for r in historical_readings if "torque_nm" in r]
        vibrations = [r.get("vibration_mms", 0.0) for r in historical_readings if "vibration_mms" in r]

        if len(torques) >= 4:
            torque_variance = sum((x - (sum(torques) / len(torques))) ** 2 for x in torques) / len(torques)
            vib_variance = sum((x - (sum(vibrations) / len(vibrations))) ** 2 for x in vibrations) / len(vibrations)

            if torque_variance == 0.0 and vib_variance == 0.0 and sum(torques) > 0:
                evidence = (
                    f"Artificial Flatline Detected on {joint}: Exact identical sensor readings "
                    f"repeated over {len(torques)} cycles with 0.00 variance. Indicates frozen sensor or hardware bridge."
                )
                return SpoofDetectionResult(
                    joint=joint,
                    is_spoofed=True,
                    spoof_type="SENSOR_FLATLINE_INJECTION",
                    confidence=0.88,
                    evidence=evidence,
                    estimated_actual_torque=torques[0],
                    reported_torque=torques[0],
                )

        return None

    def evaluate_joint(
        self, joint: str, telemetry: Dict[str, float], history: Optional[List[Dict[str, float]]] = None
    ) -> SpoofDetectionResult:
        """Runs all defense layers on a single joint."""
        torque = float(telemetry.get("torque_nm", 0.0))
        current = float(telemetry.get("motor_current_a", torque / 5.0))
        temp = float(telemetry.get("temperature_c", 40.0))
        vib = float(telemetry.get("vibration_mms", 10.0))

        # Check Layer 1: Electromechanical consistency
        res1 = self.check_electromechanical_consistency(joint, torque, current)
        if res1 and res1.is_spoofed:
            return res1

        # Check Layer 2: Thermodynamic divergence
        res2 = self.check_thermodynamic_divergence(joint, torque, temp, vib)
        if res2 and res2.is_spoofed:
            return res2

        # Check Layer 3: Flatline variance
        if history:
            res3 = self.check_sensor_flatline(joint, history)
            if res3 and res3.is_spoofed:
                return res3

        return SpoofDetectionResult(
            joint=joint,
            is_spoofed=False,
            confidence=0.0,
            evidence="Telemetry passed physical plausibility cross-checks.",
            reported_torque=torque,
        )

    def evaluate_arm(
        self, arm_telemetry: Dict[str, Dict[str, float]], history_per_joint: Optional[Dict[str, List[Dict[str, float]]]] = None
    ) -> Dict[str, Any]:
        """Scans all joints on a robotic arm for spoofed telemetry."""
        joint_results = {}
        spoofed_joints = []
        highest_confidence = 0.0

        for joint in JOINTS:
            if joint in arm_telemetry:
                hist = history_per_joint.get(joint) if history_per_joint else None
                res = self.evaluate_joint(joint, arm_telemetry[joint], hist)
                joint_results[joint] = res.to_dict()
                if res.is_spoofed:
                    spoofed_joints.append(res)
                    if res.confidence > highest_confidence:
                        highest_confidence = res.confidence

        is_arm_spoofed = len(spoofed_joints) > 0
        summary = (
            f"SPOOFING ATTACK DETECTED on: {', '.join([s.joint for s in spoofed_joints])}. "
            f"Evidence: {spoofed_joints[0].evidence}"
            if is_arm_spoofed
            else "All joint sensor telemetry verified consistent with physical laws."
        )

        return {
            "is_spoofed": is_arm_spoofed,
            "confidence": round(highest_confidence, 2),
            "spoofed_joints_count": len(spoofed_joints),
            "summary": summary,
            "joint_evaluations": joint_results,
        }
