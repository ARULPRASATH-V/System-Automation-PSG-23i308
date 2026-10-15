"""
Robotic Arm Joint Health Monitoring System — Main Pipeline Entrypoint
Demonstrates:
1. Immediate deterministic safety-stop check (Hardware cutoff, 0 ML dependence)
2. Predictive wear detection model training and inference (Historical dataset)
3. Unstructured technician maintenance notes text analysis (Second independent signal)
4. Spoofed-sensor defense (Electromechanical & thermodynamic cross-validation)
5. Decision engine producing all 4 reachable states: SAFE, MONITOR, SCHEDULE_MAINTENANCE, EMERGENCY_STOP
6. Automated generation of JSON, CSV, and Markdown audit reports
"""

import json
import csv
import sys
from pathlib import Path

# Add project root to path for direct invocation
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from assignment.src.config import (
    HISTORICAL_CSV,
    MAINTENANCE_LOG,
    REPORTS_DIR,
    DECISION_SAFE,
    DECISION_MONITOR,
    DECISION_SCHEDULE_MAINTENANCE,
    DECISION_EMERGENCY_STOP,
    ALLOWED_DECISIONS,
)
from assignment.src.predictive_model import PredictiveWearModel
from assignment.src.log_analyzer import MaintenanceLogAnalyzer
from assignment.src.spoof_defense import SpoofDefenseEngine
from assignment.src.decision_engine import ArmHealthDecisionEngine


def create_demonstration_fleet():
    """
    Constructs a fleet of test robotic arms configured to validate all 4 reachable states:
    - Arm_1: Fully nominal -> SAFE
    - Arm_2: Mild wear / minor elevated telemetry -> MONITOR
    - Arm_3: Accelerated wear + technician log notes -> SCHEDULE_MAINTENANCE
    - Arm_4: Single joint critical thermal runaway -> EMERGENCY_STOP (Deterministic)
    - Arm_5: Adversarial spoofed torque attack -> EMERGENCY_STOP (Spoof Defense)
    - Arm_6: Emergency technician inspection report -> EMERGENCY_STOP (Log Directive)
    """
    fleet = {
        "Arm_1": {
            "description": "Healthy arm with nominal telemetry and clean inspection log",
            "telemetry": {
                "shoulder": {"torque_nm": 62.0, "temperature_c": 44.0, "vibration_mms": 11.2, "motor_current_a": 12.4, "speed_rpm": 28.0, "operating_hours": 320},
                "elbow":    {"torque_nm": 42.0, "temperature_c": 41.5, "vibration_mms": 9.5,  "motor_current_a": 8.4,  "speed_rpm": 35.0, "operating_hours": 320},
                "wrist":    {"torque_nm": 16.0, "temperature_c": 37.0, "vibration_mms": 7.0,  "motor_current_a": 3.5,  "speed_rpm": 55.0, "operating_hours": 320},
            },
            "history": None,
        },
        "Arm_8": {
            "description": "Arm showing mild grease seepage advisory in technician log (MONITOR candidate)",
            "telemetry": {
                "shoulder": {"torque_nm": 68.0, "temperature_c": 48.0, "vibration_mms": 13.5, "motor_current_a": 13.6, "speed_rpm": 28.0, "operating_hours": 950},
                "elbow":    {"torque_nm": 48.0, "temperature_c": 46.0, "vibration_mms": 11.0, "motor_current_a": 9.6,  "speed_rpm": 32.0, "operating_hours": 950},
                "wrist":    {"torque_nm": 17.5, "temperature_c": 39.0, "vibration_mms": 7.8,  "motor_current_a": 3.9,  "speed_rpm": 52.0, "operating_hours": 950},
            },
            "history": None,
        },
        "Arm_2": {
            "description": "Arm with accelerated mechanical wear and technician notes of rhythmic grinding & gritty grease",
            "telemetry": {
                "shoulder": {"torque_nm": 110.0, "temperature_c": 64.0, "vibration_mms": 26.0, "motor_current_a": 22.0, "speed_rpm": 24.0, "operating_hours": 3800},
                "elbow":    {"torque_nm": 88.0,  "temperature_c": 66.0, "vibration_mms": 27.5, "motor_current_a": 17.5, "speed_rpm": 28.0, "operating_hours": 3800},
                "wrist":    {"torque_nm": 29.5,  "temperature_c": 55.0, "vibration_mms": 19.5, "motor_current_a": 6.5,  "speed_rpm": 45.0, "operating_hours": 3800},
            },
            "history": None,
        },
        "Arm_4": {
            "description": "Arm with elbow joint crossing critical temperature cutoff (88.5°C > 80°C limit)",
            "telemetry": {
                "shoulder": {"torque_nm": 75.0,  "temperature_c": 52.0, "vibration_mms": 14.0, "motor_current_a": 15.0, "speed_rpm": 28.0, "operating_hours": 2100},
                "elbow":    {"torque_nm": 78.0,  "temperature_c": 88.5, "vibration_mms": 25.0, "motor_current_a": 15.6, "speed_rpm": 30.0, "operating_hours": 2100},
                "wrist":    {"torque_nm": 18.0,  "temperature_c": 42.0, "vibration_mms": 8.0,  "motor_current_a": 4.0,  "speed_rpm": 55.0, "operating_hours": 2100},
            },
            "history": None,
        },
        "Arm_5": {
            "description": "Arm under active spoofing attack: shoulder torque clamped to 18Nm while current is 29A and vib is 42mm/s",
            "telemetry": {
                "shoulder": {"torque_nm": 18.0,  "temperature_c": 79.0, "vibration_mms": 42.0, "motor_current_a": 29.0, "speed_rpm": 25.0, "operating_hours": 4200},
                "elbow":    {"torque_nm": 65.0,  "temperature_c": 56.0, "vibration_mms": 16.0, "motor_current_a": 13.0, "speed_rpm": 32.0, "operating_hours": 4200},
                "wrist":    {"torque_nm": 19.0,  "temperature_c": 44.0, "vibration_mms": 9.0,  "motor_current_a": 4.2,  "speed_rpm": 50.0, "operating_hours": 4200},
            },
            "history": None,
        },
        "Arm_6": {
            "description": "Arm whose technician log flagged active smoke and screeching metallic binding",
            "telemetry": {
                "shoulder": {"torque_nm": 70.0,  "temperature_c": 50.0, "vibration_mms": 15.0, "motor_current_a": 14.0, "speed_rpm": 26.0, "operating_hours": 1900},
                "elbow":    {"torque_nm": 55.0,  "temperature_c": 52.0, "vibration_mms": 14.0, "motor_current_a": 11.0, "speed_rpm": 30.0, "operating_hours": 1900},
                "wrist":    {"torque_nm": 24.0,  "temperature_c": 48.0, "vibration_mms": 16.0, "motor_current_a": 5.3,  "speed_rpm": 48.0, "operating_hours": 1900},
            },
            "history": None,
        },
    }
    return fleet


def run_pipeline():
    print("=" * 80)
    print("   23IF02 SYSTEM AUTOMATION — ROBOTIC ARM HEALTH MONITORING SYSTEM")
    print("=" * 80)

    # 1. Train or Load Predictive Wear Model
    print("\n[Step 1] Initializing Predictive Wear ML Model...")
    ml_model = PredictiveWearModel()
    train_results = ml_model.train(HISTORICAL_CSV)
    print(f" -> Trained Random Forest Classifier on {train_results['sample_count']} historical records.")
    print(f" -> Test Set Accuracy: {train_results['test_accuracy'] * 100:.2f}%")
    print(f" -> 5-Fold Cross Validation Accuracy: {train_results['cv_mean_accuracy'] * 100:.2f}% (±{train_results['cv_std'] * 100:.2f}%)")
    print(f" -> Key Feature Importances: {train_results['feature_importances']}")

    # 2. Initialize Text Log Analyzer
    print("\n[Step 2] Initializing Maintenance Notes NLP Analyzer...")
    log_analyzer = MaintenanceLogAnalyzer(MAINTENANCE_LOG)
    entries = log_analyzer.parse_log_entries()
    relevant_count = sum(1 for e in entries if e["is_relevant_to_robotics"])
    print(f" -> Parsed {len(entries)} log entries ({relevant_count} arm-specific inspection notes, {len(entries) - relevant_count} filtered facility notes).")

    # 3. Initialize Spoof Defense Engine
    print("\n[Step 3] Initializing Spoofed-Sensor Defense Engine...")
    spoof_defense = SpoofDefenseEngine()
    print(" -> Configured electromechanical invariants (Kt torque-current) & thermodynamic plausibility.")

    # 4. Initialize Unified Decision Engine
    print("\n[Step 4] Running Multi-Signal Decision Engine on Fleet...")
    engine = ArmHealthDecisionEngine(
        ml_model=ml_model,
        log_analyzer=log_analyzer,
        spoof_defense=spoof_defense
    )

    fleet = create_demonstration_fleet()
    results = []
    observed_decisions = set()

    print("\n" + "-" * 80)
    print(f"{'ARM ID':<8} | {'FINAL DECISION':<22} | {'PRIMARY TRIGGER':<32} | {'ACTION'}")
    print("-" * 80)

    for arm_id, arm_data in fleet.items():
        eval_res = engine.evaluate_arm(
            arm_id=arm_id,
            arm_telemetry=arm_data["telemetry"],
            history_per_joint=arm_data["history"]
        )
        decision = eval_res["final_decision"]
        trigger = eval_res["primary_trigger"]
        action = eval_res["recommended_action"]
        observed_decisions.add(decision)

        # Truncate action for clean CLI table
        short_action = (action[:35] + "...") if len(action) > 35 else action
        print(f"{arm_id:<8} | {decision:<22} | {trigger:<32} | {short_action}")

        results.append({
            "arm_id": arm_id,
            "description": arm_data["description"],
            "decision": decision,
            "primary_trigger": trigger,
            "recommended_action": action,
            "full_evaluation": eval_res,
        })

    print("-" * 80)

    # Verify that all 4 states are reachable
    print("\n[Step 5] Validating Reachability of All 4 Required Final Decisions:")
    for dec in ALLOWED_DECISIONS:
        status = "REACHED [PASS]" if dec in observed_decisions else "MISSING [FAIL]"
        print(f" -> {dec:<22}: {status}")

    assert observed_decisions == set(ALLOWED_DECISIONS), "Not all 4 required decisions were reached!"
    print("All 4 required decisions (SAFE, MONITOR, SCHEDULE_MAINTENANCE, EMERGENCY_STOP) verified reachable.")

    # 5. Export Output Reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "arm_health_monitoring_report.json"
    csv_path = REPORTS_DIR / "arm_health_monitoring_report.csv"
    md_path = REPORTS_DIR / "example_run_summary.md"

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "fleet_size": len(fleet),
            "ml_model_accuracy": train_results["test_accuracy"],
            "cross_val_accuracy": train_results["cv_mean_accuracy"],
            "evaluations": results
        }, f, indent=2, default=str)

    # Save CSV report
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm_id", "decision", "primary_trigger", "recommended_action"])
        for r in results:
            writer.writerow([r["arm_id"], r["decision"], r["primary_trigger"], r["recommended_action"]])

    # Save Markdown Summary
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Automated Fleet Health Monitoring Report\n\n")
        f.write(f"- **ML Model Test Accuracy:** {train_results['test_accuracy'] * 100:.2f}%\n")
        f.write(f"- **5-Fold Cross Validation:** {train_results['cv_mean_accuracy'] * 100:.2f}%\n")
        f.write(f"- **Total Arms Evaluated:** {len(fleet)}\n\n")
        f.write("## Fleet Decisions Table\n\n")
        f.write("| Arm ID | Final Decision | Primary Trigger | Recommended Action |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            f.write(f"| **{r['arm_id']}** | `{r['decision']}` | {r['primary_trigger']} | {r['recommended_action']} |\n")

    print(f"\n[Step 6] Generated Output Reports:")
    print(f" -> JSON Report: {json_path}")
    print(f" -> CSV Report:  {csv_path}")
    print(f" -> MD Summary:  {md_path}")
    print("=" * 80)
    print("Execution completed successfully!")


if __name__ == "__main__":
    run_pipeline()
