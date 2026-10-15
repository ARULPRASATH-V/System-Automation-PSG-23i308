# Robotic Arm Joint Health Monitoring System
**Course:** 23IF02 · System Automation — Bonus Challenge  
**Fleet Architecture:** 6-Axis Robotic Arms (Shoulder, Elbow, Wrist Telemetry)

---

## Overview

The **Robotic Arm Joint Health Monitoring System** is an end-to-end industrial automation platform designed to monitor mechanical degradation, enforce deterministic safety cutoffs, defend against adversarial sensor spoofing, and synthesize qualitative technician logs with quantitative machine learning predictions.

The system produces exactly **one final operational decision per robotic arm**:
1. `SAFE` — Nominal operation.
2. `MONITOR` — Early wear trend or minor advisory observation.
3. `SCHEDULE_MAINTENANCE` — Accelerated bearing/gear degradation or technician defect report.
4. `EMERGENCY_STOP` — Immediate deterministic physical threshold breach, critical spoof attack, or active safety hazard directive.

---

## Deliverables & Project Structure

```
assignment/
├── data/
│   ├── historical_sensor_readings.csv   # 150 labeled historical sensor records (> 50 required)
│   ├── generate_synthetic_data.py       # Deterministic dataset generator
│   └── maintenance_log.txt              # 16-entry shared log mixing relevant and irrelevant notes
├── src/
│   ├── __init__.py
│   ├── config.py                        # Hardware limits, Kt constants, NLP taxonomy
│   ├── safety_guard.py                  # Req 1: Deterministic safety-stop (Zero ML dependence)
│   ├── predictive_model.py              # Req 2: Random Forest joint wear classifier
│   ├── log_analyzer.py                  # Req 3: Unstructured text maintenance log analyzer
│   ├── spoof_defense.py                 # Req 4: Multi-layered physical spoof defense
│   └── decision_engine.py               # Req 5: Multi-signal arbitration engine (All 4 states reachable)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Test path configuration
│   ├── test_safety_guard.py             # Unit tests for hardware cutoffs & fail-safes
│   ├── test_predictive_model.py         # Unit tests for ML dataset, training, inference
│   ├── test_log_analyzer.py             # Unit tests for NLP filtering, scoring, negation
│   ├── test_spoof_defense.py            # Unit tests for Kt mismatch, thermal divergence, flatlines
│   ├── test_decision_engine.py          # Unit tests for reachability & priority hierarchy
│   └── test_integration.py             # End-to-end integration test
├── reports/
│   ├── arm_health_monitoring_report.json # Structured machine-readable audit report
│   ├── arm_health_monitoring_report.csv  # Fleet decision tabular summary
│   └── example_run_summary.md           # Markdown executive summary
├── main.py                              # Main executable demonstration pipeline
├── EXPLANATION.md                       # Comprehensive half-page technical explanation
├── README.md                            # Installation and execution instructions
└── SUBMISSION.md                        # Submission checklist and requirement mapping
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `pip` package manager

### Environment Installation
Run the following from the repository root:
```bash
pip install -r requirements.txt
```

Core dependencies:
- `numpy>=1.26.0`
- `pandas>=2.2.0`
- `scikit-learn>=1.4.0`
- `joblib>=1.3.0`
- `pytest>=8.0.0`

---

## Execution Instructions

### 1. Run the Main Demonstration Pipeline
Executes model training, log analysis, spoof checks, fleet arbitration, and report generation:
```bash
python assignment/main.py
```

### 2. Run the Automated Test Suite
Runs all 26 unit and integration tests with detailed output:
```bash
python -m pytest assignment/tests/ -v
```

### 3. Regenerate Synthetic Dataset (Optional)
```bash
python assignment/data/generate_synthetic_data.py
```

---

## System Architecture & Core Capabilities

### 1. Deterministic Immediate Safety Stop (Requirement 1)
- **Constraint:** Strictly independent of the ML model.
- Evaluates raw telemetry against hard physical limits:
  - **Shoulder:** Torque $> 180\,\text{Nm}$, Temp $> 85^\circ\text{C}$, Vibration $> 45\,\text{mm/s}$
  - **Elbow:** Torque $> 120\,\text{Nm}$, Temp $> 80^\circ\text{C}$, Vibration $> 38\,\text{mm/s}$
  - **Wrist:** Torque $> 45\,\text{Nm}$, Temp $> 75^\circ\text{C}$, Vibration $> 30\,\text{mm/s}$
- Sensor dropouts (`None`, `NaN`, `inf`) trigger an immediate fail-safe hold.

### 2. Predictive Wear Detection (Requirement 2)
- Trained on 150 labeled historical records across `NORMAL`, `EARLY_WEAR`, `ACCELERATED_WEAR`, and `CRITICAL_WEAR`.
- Random Forest model achieves $> 97\%$ test accuracy and $> 98\%$ 5-fold cross-validation accuracy.
- Computes wear probability distribution and normalized risk score $[0.0\text{–}1.0]$.

### 3. Maintenance Notes NLP (Requirement 3)
- Ingests unstructured free-text inspection notes.
- Automatically identifies mentioned arms (`Arm_1`, `Arm_2`, etc.) and joints (`shoulder`, `elbow`, `wrist`).
- Filters out irrelevant facility chatter (coffee overflows, Wi-Fi upgrades, lunch notices).
- Evaluates negation syntax ("zero play", "no heat") to avoid false positives.

### 4. Spoofed-Sensor Defense (Requirement 4)
- **Electromechanical Coupling:** $\tau_{expected} = K_t \cdot I_{motor}$. Detects when reported torque is artificially clamped low while motor current draws heavy overload power ($> 70\%$ divergence).
- **Thermodynamic Invariance:** Prevents adversaries from masking severe friction by catching thermal ($> 75^\circ\text{C}$) and vibration ($> 35\,\text{mm/s}$) surges during claimed low-torque operation.
- **Flatline Injection Check:** Catches frozen static signals with zero ADC variance.

### 5. Multi-Signal Decision Matrix (Requirement 5)
- Prioritizes deterministic safety and high-confidence spoof alarms first.
- Arbitrates subtle mechanical wear against technician notes for scheduled PM.
- Demonstrates **all four reachable states** (`SAFE`, `MONITOR`, `SCHEDULE_MAINTENANCE`, `EMERGENCY_STOP`).

---

## Output Reports Generated

Upon execution of `python assignment/main.py`, reports are automatically generated in `assignment/reports/`:
- `arm_health_monitoring_report.json` — Detailed JSON object with full telemetry, ML probabilities, and reasons.
- `arm_health_monitoring_report.csv` — Tabular CSV summary.
- `example_run_summary.md` — Markdown summary table of fleet health.
