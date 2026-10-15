"""
Requirement 5: Multi-Signal Decision Engine

Synthesizes:
1. Deterministic Safety Guard (Hardware cutoff)
2. Spoofed-Sensor Defense (Physical plausibility)
3. Predictive Wear ML Model (Early wear detection)
4. Maintenance Log NLP (Unstructured technician notes)

Produces exactly ONE final decision per robotic arm:
- SAFE
- MONITOR
- SCHEDULE_MAINTENANCE
- EMERGENCY_STOP
"""

from typing import Dict, Any, List, Optional
from assignment.src.config import (
    DECISION_SAFE,
    DECISION_MONITOR,
    DECISION_SCHEDULE_MAINTENANCE,
    DECISION_EMERGENCY_STOP,
    ALLOWED_DECISIONS,
)
from assignment.src.safety_guard import check_arm_safety_stop, check_arm_warnings
from assignment.src.predictive_model import PredictiveWearModel
from assignment.src.log_analyzer import MaintenanceLogAnalyzer
from assignment.src.spoof_defense import SpoofDefenseEngine


class ArmHealthDecisionEngine:
    def __init__(
        self,
        ml_model: Optional[PredictiveWearModel] = None,
        log_analyzer: Optional[MaintenanceLogAnalyzer] = None,
        spoof_defense: Optional[SpoofDefenseEngine] = None,
    ):
        self.ml_model = ml_model if ml_model is not None else PredictiveWearModel()
        self.log_analyzer = log_analyzer if log_analyzer is not None else MaintenanceLogAnalyzer()
        self.spoof_defense = spoof_defense if spoof_defense is not None else SpoofDefenseEngine()

        # Attempt to load ML model if already trained
        if self.ml_model.model is None:
            self.ml_model.load()

    def evaluate_arm(
        self,
        arm_id: str,
        arm_telemetry: Dict[str, Dict[str, float]],
        history_per_joint: Optional[Dict[str, List[Dict[str, float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a robotic arm and produces one final operational decision.
        """
        reasons = []
        action_recommendation = ""

        # -------------------------------------------------------------
        # STEP 1: DETERMINISTIC IMMEDIATE SAFETY STOP CHECK (Priority 1)
        # CRITICAL CONSTRAINT: Must NEVER depend on the ML model.
        # -------------------------------------------------------------
        is_hard_stop, safety_violations = check_arm_safety_stop(arm_telemetry)
        if is_hard_stop:
            violation_msgs = [v["message"] for v in safety_violations]
            return {
                "arm_id": arm_id,
                "final_decision": DECISION_EMERGENCY_STOP,
                "primary_trigger": "DETERMINISTIC_SAFETY_VIOLATION",
                "reasons": violation_msgs,
                "recommended_action": "IMMEDIATE POWER CUT. Stop arm motion instantly to prevent physical destruction.",
                "telemetry_summary": arm_telemetry,
                "safety_violations": safety_violations,
                "spoof_defense": None,
                "ml_prediction": None,
                "log_analysis": None,
            }

        # -------------------------------------------------------------
        # STEP 2: SPOOFED-SENSOR DEFENSE EVALUATION (Priority 2)
        # Cross-sensor physical plausibility validation
        # -------------------------------------------------------------
        spoof_report = self.spoof_defense.evaluate_arm(arm_telemetry, history_per_joint)
        if spoof_report["is_spoofed"] and spoof_report["confidence"] >= 0.70:
            return {
                "arm_id": arm_id,
                "final_decision": DECISION_EMERGENCY_STOP,
                "primary_trigger": "CRITICAL_SPOOFED_SENSOR_ATTACK",
                "reasons": [spoof_report["summary"]],
                "recommended_action": (
                    "EMERGENCY HALT & SECURITY AUDIT: Sensor telemetry has been maliciously falsified or "
                    "dangerously compromised. Halt arm and recalibrate sensor hardware."
                ),
                "telemetry_summary": arm_telemetry,
                "safety_violations": [],
                "spoof_defense": spoof_report,
                "ml_prediction": None,
                "log_analysis": None,
            }

        # -------------------------------------------------------------
        # STEP 3: MAINTENANCE NOTES LOG ANALYSIS (Signal 2)
        # -------------------------------------------------------------
        log_report = self.log_analyzer.analyze_arm_notes(arm_id)
        if log_report["has_notes"] and log_report["log_risk_level"] == "CRITICAL":
            return {
                "arm_id": arm_id,
                "final_decision": DECISION_EMERGENCY_STOP,
                "primary_trigger": "TECHNICIAN_LOG_EMERGENCY_DIRECTIVE",
                "reasons": [log_report["summary"]],
                "recommended_action": "EMERGENCY INSPECTION REQUIRED: Active hazard reported in technician log.",
                "telemetry_summary": arm_telemetry,
                "safety_violations": [],
                "spoof_defense": spoof_report,
                "ml_prediction": None,
                "log_analysis": log_report,
            }

        # -------------------------------------------------------------
        # STEP 4: PREDICTIVE WEAR MODEL EVALUATION (Signal 1)
        # -------------------------------------------------------------
        ml_report = self.ml_model.evaluate_arm(arm_telemetry)

        # -------------------------------------------------------------
        # STEP 5: SENSOR WARNINGS CHECK (Degradation signals)
        # -------------------------------------------------------------
        sensor_warnings = check_arm_warnings(arm_telemetry)

        # -------------------------------------------------------------
        # STEP 6: SIGNAL COMBINATION AND DECISION ARBITRATION
        # Combine ML Wear, Log Analysis, Warnings, and Suspicious Spoofing
        # -------------------------------------------------------------
        ml_risk = ml_report["max_wear_risk_score"]
        ml_state = ml_report["arm_wear_state"]
        log_risk = log_report["log_risk_score"]
        log_level = log_report["log_risk_level"]
        has_sensor_warnings = len(sensor_warnings) > 0
        has_moderate_spoof_suspicion = spoof_report["is_spoofed"] and spoof_report["confidence"] < 0.70

        # Scenario A: SCHEDULE_MAINTENANCE
        # Triggered by:
        # - ML predicts ACCELERATED_WEAR or CRITICAL_WEAR (or ml_risk >= 0.65)
        # - High-severity maintenance note (log_risk >= 0.65, e.g., severe grinding, mechanical backlash)
        # - Coincident warning signals (ml_risk >= 0.40 and log_risk >= 0.40)
        # - Multiple simultaneous sensor warning violations
        # - Moderate spoof suspicion requiring maintenance bench test
        is_high_ml_wear = ml_state in ["ACCELERATED_WEAR", "CRITICAL_WEAR"] or ml_risk >= 0.65
        is_severe_log_issue = log_risk >= 0.65
        is_coincident_degradation = (ml_risk >= 0.35 and log_risk >= 0.35)
        is_multiple_warnings = len(sensor_warnings) >= 2

        if (
            is_high_ml_wear
            or is_severe_log_issue
            or is_coincident_degradation
            or is_multiple_warnings
            or has_moderate_spoof_suspicion
        ):
            triggers = []
            if is_high_ml_wear:
                triggers.append(f"ML Model detected {ml_state} (risk: {ml_risk:.2f}) on {ml_report['worst_joint']}")
            if is_severe_log_issue:
                triggers.append(f"Technician log reported significant defect: {log_report['summary']}")
            if is_coincident_degradation:
                triggers.append(f"Coincident degradation: ML risk ({ml_risk:.2f}) + Log risk ({log_risk:.2f})")
            if is_multiple_warnings:
                triggers.append(f"Multiple sensor warnings: {[w['message'] for w in sensor_warnings]}")
            if has_moderate_spoof_suspicion:
                triggers.append(f"Electromechanical discrepancy flagged: {spoof_report['summary']}")

            return {
                "arm_id": arm_id,
                "final_decision": DECISION_SCHEDULE_MAINTENANCE,
                "primary_trigger": "MECHANICAL_DEGRADATION_OR_LOG_WARNING",
                "reasons": triggers,
                "recommended_action": "Schedule physical inspection, lubricate bearings, and replace degraded components during next maintenance window.",
                "telemetry_summary": arm_telemetry,
                "safety_violations": [],
                "sensor_warnings": sensor_warnings,
                "spoof_defense": spoof_report,
                "ml_prediction": ml_report,
                "log_analysis": log_report,
            }

        # Scenario B: MONITOR
        # Triggered by:
        # - ML predicts EARLY_WEAR (or 0.20 <= ml_risk < 0.65)
        # - Mild log observation (0 < log_risk < 0.65, e.g., seepage, slight play, monitoring advised)
        # - Single sensor metric elevated above warning threshold
        is_early_ml_wear = (ml_state == "EARLY_WEAR") or (0.20 <= ml_risk < 0.65)
        is_mild_log_note = (0.0 < log_risk < 0.65)
        is_single_warning = (len(sensor_warnings) == 1)

        if is_early_ml_wear or is_mild_log_note or is_single_warning:
            triggers = []
            if is_early_ml_wear:
                triggers.append(f"ML Model detected early wear trend (state: {ml_state}, risk: {ml_risk:.2f})")
            if is_mild_log_note:
                triggers.append(f"Maintenance note advisory: {log_report['summary']}")
            if is_single_warning:
                triggers.append(f"Elevated sensor metric: {sensor_warnings[0]['message']}")

            return {
                "arm_id": arm_id,
                "final_decision": DECISION_MONITOR,
                "primary_trigger": "EARLY_WEAR_OR_ELEVATED_METRIC",
                "reasons": triggers,
                "recommended_action": "Increase telemetry sampling rate to 100 Hz, track thermal trend, and review at end-of-shift.",
                "telemetry_summary": arm_telemetry,
                "safety_violations": [],
                "sensor_warnings": sensor_warnings,
                "spoof_defense": spoof_report,
                "ml_prediction": ml_report,
                "log_analysis": log_report,
            }

        # Scenario C: SAFE
        # All checks passed, low risk, clean logs
        return {
            "arm_id": arm_id,
            "final_decision": DECISION_SAFE,
            "primary_trigger": "ALL_NOMINAL",
            "reasons": ["All joint sensors within normal operating limits", "ML predicts healthy nominal state", "No negative maintenance notes"],
            "recommended_action": "Continue normal assembly line operations.",
            "telemetry_summary": arm_telemetry,
            "safety_violations": [],
            "sensor_warnings": [],
            "spoof_defense": spoof_report,
            "ml_prediction": ml_report,
            "log_analysis": log_report,
        }
