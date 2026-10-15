# Engineering Explanation & Architectural Design Document
**Project:** Robotic Arm Joint Health Monitoring System (6-Axis Fleet)  
**Course:** 23IF02 · System Automation — Bonus Challenge  

---

## 1. Safety Threshold Selection for Immediate Safety Stop (Deterministic Guard)

In high-speed 6-axis industrial robotics, joint mechanical failures and thermal runaways manifest with extreme non-linear acceleration. Waiting for batch feature extraction, model inference, or asynchronous network calls introduces lethal latency ($\ge 50\text{–}200\,\text{ms}$) that can shatter harmonic drive gearing, fracture cast aluminum arm castings, or injure line operators. Therefore, the immediate safety-stop system is strictly deterministic, executing in sub-millisecond $O(1)$ time with **zero dependency on machine learning**.

The critical thresholds were chosen based on standard industrial robot mechanical limits (e.g., ISO 10218 safety requirements for industrial robots and manufacturer gearbox duty ratings):

1. **Joint Torque Thresholds:**
   - **Shoulder (J1/J2):** $180.0\,\text{Nm}$ cutoff (continuous rating: $110\,\text{Nm}$, peak structural limit: $200\,\text{Nm}$). Exceeding $180\,\text{Nm}$ indicates a physical collision, severe payload overload, or mechanical jam.
   - **Elbow (J3):** $120.0\,\text{Nm}$ cutoff (continuous rating: $70\,\text{Nm}$). Exceeding this limit threatens mid-arm casting shear and cross-roller bearing deformation.
   - **Wrist (J4–J6):** $45.0\,\text{Nm}$ cutoff (continuous rating: $25\,\text{Nm}$). Wrist precision planetary and cycloidal gear teeth fracture under shock torques $> 50\,\text{Nm}$.
2. **Joint Temperature Thresholds:**
   - **Shoulder & Elbow:** $85.0^\circ\text{C}$ and $80.0^\circ\text{C}$ cutoffs. Standard Class F servo motor winding insulation degrades rapidly above $100^\circ\text{C}$, and synthetic bearing grease experiences thermal breakdown and oil separation above $85^\circ\text{C}$.
   - **Wrist:** $75.0^\circ\text{C}$ cutoff. Compact harmonic drive flexsplines suffer accelerated tooth fatigue and thermal binding when operating above $75^\circ\text{C}$.
3. **Vibration Thresholds:**
   - **Shoulder, Elbow, Wrist:** $45.0\,\text{mm/s}$, $38.0\,\text{mm/s}$, and $30.0\,\text{mm/s}$ RMS velocity cutoffs (ISO 10816 Class II/III machine vibration standards). Vibration exceeding these levels indicates raceway spalling, ball bearing cage collapse, or structural bolt loosening.
4. **Sensor Fault Fail-Safe:**
   - Any dropout, null (`None`), infinite value (`inf`), or `NaN` trips an immediate `EMERGENCY_STOP` (`CRITICAL_SENSOR_FAULT`) to guarantee fail-safe behavior under wire shearing.

---

## 2. Predictive Modeling & Unstructured Maintenance Log Combination Logic

The system utilizes a dual-signal decision pipeline combining quantitative telemetry predictions with qualitative human observations:

```
+-----------------------------------------------------------------------------------+
| Telemetry Input                                                                   |
| [Torque, Temp, Vibration, Current, Speed, Hours]                                  |
+-----------------------------------------------------------------------------------+
       |                                                    |
       v                                                    v
[Deterministic Safety Guard]                 [Spoofed-Sensor Defense Engine]
(Hardware Threshold Check)                   (Electromechanical & Thermodynamic Invariants)
       |                                                    |
   Violated? ---> YES ---> EMERGENCY_STOP               Spoofed? ---> YES ---> EMERGENCY_STOP
       | (NO)                                               | (NO)
       +-------------------------+--------------------------+
                                 |
                                 v
               +-----------------------------------+
               | Dual-Signal Arbitration Engine     |
               +-----------------------------------+
                 /                               \
                v                                 v
     [Signal 1: ML Model]               [Signal 2: NLP Log Analyzer]
     Random Forest Classifier           Lexicon & Negation Parser
     - Feature Extraction               - Filters Irrelevant Notes
     - Wear State: Normal / Early /     - Scores Severity: Normal /
       Accelerated / Critical             Advisory / Caution / Critical
     - Wear Risk Score: [0.0 - 1.0]     - Log Risk Score: [0.0 - 1.0]
                 \                               /
                  +--------------+---------------+
                                 |
                                 v
                   +---------------------------+
                   | Final Multi-Signal Matrix |
                   +---------------------------+
                   | - EMERGENCY_STOP          |
                   | - SCHEDULE_MAINTENANCE    |
                   | - MONITOR                 |
                   | - SAFE                    |
                   +---------------------------+
```

### Signal Combination Matrix:
1. **Emergency Priority (Safety Stop & Critical Directive):**
   - If Safety Guard flags a critical hardware threshold $\rightarrow$ `EMERGENCY_STOP`.
   - If Spoof Defense detects high-confidence falsification $\rightarrow$ `EMERGENCY_STOP`.
   - If Maintenance Log records an active physical emergency (e.g., "smoke smell", "screeching binding", "immediate shutdown advised") $\rightarrow$ `EMERGENCY_STOP`.
2. **Maintenance Priority (`SCHEDULE_MAINTENANCE`):**
   - Triggered when ML predicts `ACCELERATED_WEAR` or `CRITICAL_WEAR` ($\text{ML Risk} \ge 0.65$), OR
   - When Maintenance Log reports a severe mechanical defect ($\text{Log Risk} \ge 0.65$, e.g., "grinding noise", "mechanical backlash", "discolored gritty grease"), OR
   - When coincident moderate signals occur ($\text{ML Risk} \ge 0.35$ **AND** $\text{Log Risk} \ge 0.35$), OR
   - When multiple sensor warning thresholds are exceeded simultaneously.
3. **Observation Priority (`MONITOR`):**
   - Triggered when ML predicts `EARLY_WEAR` ($0.20 \le \text{ML Risk} < 0.65$), OR
   - When Maintenance Log contains mild advisories ($0.0 < \text{Log Risk} < 0.65$, e.g., "mild grease seepage", "monitoring advised" after filtering negations like "zero play"), OR
   - When a single sensor metric exceeds warning boundaries while all others remain nominal.
4. **Baseline Nominal (`SAFE`):**
   - All joint metrics well within standard envelope, ML predicts `NORMAL`, no negative maintenance notes logged.

---

## 3. Spoofed-Sensor Defense Mechanics: Capabilities and Limitations

### How It Works
Adversaries or compromised sensor transmitters may deliberately inject false low-torque readings (e.g., reporting $18\,\text{Nm}$ when true torque is $140\,\text{Nm}$) to circumvent automated safety shutdowns and induce catastrophic physical crashes. Our defense operates on **first-principles physical invariants**:

1. **Electromechanical Coupling Invariant ($\tau = K_t \cdot I_{motor}$):**
   The electromagnetic torque generated by a brushless DC / PMSM motor is fundamentally proportional to the motor stator current $I$ via torque constant $K_t$. If an attacker alters the digital torque reading to $18\,\text{Nm}$ while current sensors draw $29.0\,\text{A}$ ($K_t = 5.0\,\text{Nm/A} \implies \tau_{physical} \approx 145\,\text{Nm}$), the relative divergence exceeds $80\%$. Divergence $> 40\%$ raises a warning; divergence $> 70\%$ immediately trips a spoof detection emergency halt.
2. **Thermodynamic & Kinetic Dissipation Invariance:**
   By the conservation of energy, high joint temperature ($> 75^\circ\text{C}$) and severe mechanical vibration ($> 35\,\text{mm/s}$) are thermodynamic dissipations of friction work ($P = \tau \cdot \omega$). A joint idling at $< 20\,\text{Nm}$ cannot physically produce high heat and severe vibration unless ambient temperature is extreme. When thermal/vibration energy contradicts claimed torque, a thermodynamic divergence violation is flagged.
3. **Sensor Flatline / Zero-Variance Detection:**
   Real physical sensors exhibit stochastic ADC quantization noise and mechanical ripple ($\sigma > 0.05\,\text{mm/s}$). Clamped or replay spoofing inputs that transmit static constants ($0.00$ variance over 4 consecutive cycles) are flagged as artificial signal injections.

### What Attacks It Catches
- **Torque Under-Reporting / Masking Attacks:** Attackers feeding low torque values to hide gearbox binding, overloaded payload, or seized bearings.
- **Constant Clamping / Resistor Bridge Spoofs:** Replay or voltage-clamp attacks outputting static values with zero electronic noise.
- **Single-Sensor Man-in-the-Middle (MitM):** Attackers compromising only the CAN bus ID or ADC channel of the torque sensor without simultaneously falsifying motor drives and accelerometers.

### What Attacks It Would NOT Catch
- **Coordinated Multi-Sensor Collusion Attack:** If a sophisticated nation-state adversary simultaneously compromises the motor current drive firmware, torque strain gauge ADC, thermocouple amplifier, and vibration accelerometer, and computes a physically self-consistent fake state vector $(\tau, I, T, \text{vib})$, physics-based cross-validation alone cannot differentiate it from actual operation without cryptographic bus signing (e.g., CAN FD SecOC / hardware TPM authentication).
- **Sub-Threshold Gradual Drift Injection:** An attacker injecting small, slow torque biases within the allowable electromechanical tolerance band ($\le 30\%$ deviation) during light dynamic motion, which mimics normal friction variability.
