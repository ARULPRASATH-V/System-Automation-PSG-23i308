"""
Unit tests for Requirement 3: Maintenance Notes NLP Analyzer
"""

import pytest
from assignment.src.config import MAINTENANCE_LOG
from assignment.src.log_analyzer import MaintenanceLogAnalyzer


def test_maintenance_log_has_at_least_8_entries():
    analyzer = MaintenanceLogAnalyzer(MAINTENANCE_LOG)
    entries = analyzer.parse_log_entries()
    assert len(entries) >= 8, f"Expected at least 8 entries, got {len(entries)}"


def test_irrelevant_notes_filtered():
    analyzer = MaintenanceLogAnalyzer(MAINTENANCE_LOG)
    entries = analyzer.parse_log_entries()
    irrelevant = [e for e in entries if not e["is_relevant_to_robotics"]]
    # Facility notes like coffee machine, wifi, fire extinguisher should be flagged irrelevant
    assert len(irrelevant) >= 4
    texts = " ".join([e["raw_text"].lower() for e in irrelevant])
    assert "coffee" in texts
    assert "wifi" in texts


def test_emergency_inspection_scoring():
    analyzer = MaintenanceLogAnalyzer()
    critical_text = "Arm_6 wrist joint suffered sudden severe mechanical binding. Screeching metallic sound, smoke smell. Immediate shutdown advised!"
    score = analyzer.score_entry_severity(critical_text)
    assert score["level"] == "CRITICAL"
    assert score["score"] == 1.0
    assert any(k in score["keywords"] for k in ["smoke", "screeching", "shutdown", "binding"])


def test_caution_inspection_scoring():
    analyzer = MaintenanceLogAnalyzer()
    warn_text = "Noticed Arm_2 elbow joint emitting rhythmic grinding noise. Grease discolored."
    score = analyzer.score_entry_severity(warn_text)
    assert score["level"] == "CAUTION"
    assert score["score"] >= 0.65
    assert "grinding" in score["keywords"]


def test_negation_handling():
    analyzer = MaintenanceLogAnalyzer()
    neg_text = "Arm_1 running smooth, zero excessive heat or play detected, no vibration."
    score = analyzer.score_entry_severity(neg_text)
    assert "play" not in score["keywords"]
    assert "vibration" not in score["keywords"]
    assert score["level"] == "HEALTHY"


def test_arm_specific_aggregation():
    analyzer = MaintenanceLogAnalyzer()
    report = analyzer.analyze_arm_notes("Arm_2")
    assert report["has_notes"] is True
    assert report["log_risk_level"] == "CAUTION"
    assert "elbow" in report["mentioned_joints"]
    assert "grinding" in report["all_keywords"]
