"""
End-to-End Integration Tests
Validates pipeline execution and report file artifact creation.
"""

import json
import csv
from pathlib import Path
from assignment.main import run_pipeline
from assignment.src.config import REPORTS_DIR


def test_full_pipeline_run_and_report_generation():
    # Run the entire pipeline
    run_pipeline()

    json_report = REPORTS_DIR / "arm_health_monitoring_report.json"
    csv_report = REPORTS_DIR / "arm_health_monitoring_report.csv"
    md_report = REPORTS_DIR / "example_run_summary.md"

    assert json_report.exists(), "JSON report must exist."
    assert csv_report.exists(), "CSV report must exist."
    assert md_report.exists(), "Markdown report must exist."

    # Validate JSON content
    with open(json_report, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "fleet_size" in data
        assert data["fleet_size"] >= 4
        assert "evaluations" in data
        decisions = {e["decision"] for e in data["evaluations"]}
        assert {"SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP"}.issubset(decisions)

    # Validate CSV content
    with open(csv_report, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) >= 4
        assert "arm_id" in rows[0]
        assert "decision" in rows[0]
