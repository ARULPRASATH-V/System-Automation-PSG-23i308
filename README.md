# 23IF02 System Automation — Robotic Arm Health Monitoring System
**Student / Submitter:** ARULPRASATH-V  
**Repository:** [ARULPRASATH-V/System-Automation-PSG-23i308](https://github.com/ARULPRASATH-V/System-Automation-PSG-23i308)

---

## Project Structure

This repository contains workshop exercises and the complete submission-ready project for the **Robotic Arm Joint Health Monitoring System (Bonus Challenge)** located in the [`assignment/`](assignment/) directory:

- [`assignment/`](assignment/) — **Complete Bonus Challenge Submission Folder**
  - [`main.py`](assignment/main.py) — Pipeline CLI demonstrating all 4 reachable states (`SAFE`, `MONITOR`, `SCHEDULE_MAINTENANCE`, `EMERGENCY_STOP`)
  - [`src/`](assignment/src/) — Source code (`safety_guard.py`, `predictive_model.py`, `log_analyzer.py`, `spoof_defense.py`, `decision_engine.py`, `config.py`)
  - [`data/`](assignment/data/) — Synthetic dataset with 150 labeled rows & 16-entry maintenance log
  - [`tests/`](assignment/tests/) — Comprehensive pytest test suite (26 passing tests)
  - [`reports/`](assignment/reports/) — Output reports in JSON, CSV, and Markdown
  - [`EXPLANATION.md`](assignment/EXPLANATION.md) — Technical explanation document
  - [`SUBMISSION.md`](assignment/SUBMISSION.md) — Deliverables checklist and requirements mapping
  - [`README.md`](assignment/README.md) — Detailed assignment documentation
- [`Day1/`](Day1/) — Day 1 Rule Engine exercises and logs
- [`Day2/`](Day2/) — Day 2 Predictive maintenance models and experiments
- [`requirements.txt`](requirements.txt) — Python dependencies

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Assignment Pipeline
```bash
python assignment/main.py
```

### 3. Run Automated Tests
```bash
python -m pytest assignment/tests/ -v
```

See [`SUBMISSION.md`](SUBMISSION.md) for the complete checklist and verification summary.
