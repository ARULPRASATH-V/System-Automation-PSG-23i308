"""
Requirement 3: Maintenance Notes Analyzer (Unstructured Text Signal)

Extracts and synthesizes unstructured free-text inspection notes into an
independent risk signal per robotic arm.
Filters out irrelevant facility notes (coffee spills, Wi-Fi reboots, lamps, etc.).
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from assignment.src.config import MAINTENANCE_LOG, LOG_KEYWORDS, JOINTS


class MaintenanceLogAnalyzer:
    def __init__(self, log_path: Path = MAINTENANCE_LOG):
        self.log_path = Path(log_path)
        self.critical_keywords = [k.lower() for k in LOG_KEYWORDS["CRITICAL"]]
        self.warning_keywords = [k.lower() for k in LOG_KEYWORDS["WARNING"]]
        self.healthy_keywords = [k.lower() for k in LOG_KEYWORDS["HEALTHY"]]
        self.irrelevant_topics = [k.lower() for k in LOG_KEYWORDS["IRRELEVANT_TOPICS"]]

    def parse_log_entries(self) -> List[Dict[str, Any]]:
        """Parses the text log into structured entries."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Maintenance log not found at {self.log_path}")

        entries = []
        pattern = re.compile(r"^\[(.*?)\]\s*(.*)$")

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                match = pattern.match(line)
                if match:
                    timestamp, text = match.groups()
                else:
                    timestamp = "UNKNOWN"
                    text = line

                # Arm entity detection: e.g. "Arm_1", "Arm-2", "arm 3"
                arm_match = re.search(r"\b[Aa]rm[_\-\s]?(\d+)\b", text)
                arm_id = f"Arm_{arm_match.group(1)}" if arm_match else None

                # Joint detection
                text_lower = text.lower()
                detected_joints = [j for j in JOINTS if j in text_lower]

                # Check relevance
                has_arm = arm_id is not None
                is_irrelevant_topic = any(irr in text_lower for irr in self.irrelevant_topics)
                is_relevant = has_arm or not is_irrelevant_topic

                entries.append({
                    "line_number": line_no,
                    "timestamp": timestamp,
                    "raw_text": text,
                    "arm_id": arm_id,
                    "joints": detected_joints,
                    "is_relevant_to_robotics": is_relevant and has_arm,
                })

        return entries

    def score_entry_severity(self, text: str) -> Dict[str, Any]:
        """Scores a single text passage based on domain lexicon with negation handling."""
        text_lower = text.lower()

        # Check for negation patterns (e.g. "zero play", "no heat", "without vibration")
        negation_pattern = re.compile(r"\b(no|zero|without|never|neither|nor)\b(?:\s+\w+){0,3}\s+\b(\w+)\b")
        negated_words = set()
        for match in negation_pattern.finditer(text_lower):
            negated_words.add(match.group(2))

        matched_crit = [k for k in self.critical_keywords if k in text_lower and k not in negated_words]
        matched_warn = [k for k in self.warning_keywords if k in text_lower and k not in negated_words]
        matched_health = [k for k in self.healthy_keywords if k in text_lower]

        if matched_crit:
            return {
                "level": "CRITICAL",
                "score": 1.0,
                "keywords": matched_crit,
                "reason": f"Critical indicators observed: {', '.join(matched_crit)}",
            }
        elif matched_warn:
            # Differentiate between severe mechanical defects (grinding, backlash) vs mild advisory (seepage, mild)
            is_mild = any(w in text_lower for w in ["mild", "recommend monitoring", "minor", "slight"])
            if is_mild and not any(w in matched_warn for w in ["grinding", "backlash", "jitter"]):
                score = 0.45
                level = "ADVISORY"
            else:
                score = min(0.65 + (0.10 * len(matched_warn)), 0.90)
                level = "CAUTION"

            return {
                "level": level,
                "score": round(score, 2),
                "keywords": matched_warn,
                "reason": f"Warning indicators observed: {', '.join(matched_warn)}",
            }
        elif matched_health:
            return {
                "level": "HEALTHY",
                "score": 0.0,
                "keywords": matched_health,
                "reason": "Routine / clean inspection validated.",
            }
        else:
            return {
                "level": "NEUTRAL",
                "score": 0.1,
                "keywords": [],
                "reason": "No strong mechanical sentiment detected.",
            }

    def analyze_arm_notes(self, arm_id: str) -> Dict[str, Any]:
        """
        Synthesizes all log entries mentioning a specific arm into an independent signal.
        """
        all_entries = self.parse_log_entries()
        arm_entries = [e for e in all_entries if e["arm_id"] == arm_id and e["is_relevant_to_robotics"]]

        if not arm_entries:
            return {
                "arm_id": arm_id,
                "has_notes": False,
                "log_risk_level": "HEALTHY",
                "log_risk_score": 0.0,
                "mentioned_joints": [],
                "all_keywords": [],
                "relevant_entries_count": 0,
                "summary": "No technician notes logged for this arm.",
            }

        max_score = 0.0
        worst_level = "HEALTHY"
        all_keywords = set()
        all_joints = set()
        reasons = []

        for entry in arm_entries:
            scoring = self.score_entry_severity(entry["raw_text"])
            all_keywords.update(scoring["keywords"])
            all_joints.update(entry["joints"])
            reasons.append(scoring["reason"])

            if scoring["score"] > max_score:
                max_score = scoring["score"]
                worst_level = scoring["level"]

        return {
            "arm_id": arm_id,
            "has_notes": True,
            "log_risk_level": worst_level,
            "log_risk_score": round(max_score, 2),
            "mentioned_joints": list(all_joints),
            "all_keywords": list(all_keywords),
            "relevant_entries_count": len(arm_entries),
            "summary": "; ".join(reasons),
            "latest_entry": arm_entries[-1]["raw_text"],
        }
