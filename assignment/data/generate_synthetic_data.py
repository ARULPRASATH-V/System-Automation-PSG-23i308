"""
Synthetic Historical Dataset Generator
Generates labeled historical sensor readings for 6-axis robotic arm joint health monitoring.
Features:
- Joint types: shoulder, elbow, wrist
- Sensors: torque (Nm), temperature (°C), vibration (mm/s), motor_current (A), speed_rpm (RPM)
- Labels: NORMAL, EARLY_WEAR, ACCELERATED_WEAR, CRITICAL_WEAR
"""

import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "historical_sensor_readings.csv"

# Physical operating baselines per joint
JOINT_SPECS = {
    "shoulder": {
        "torque_base": (40.0, 95.0),
        "temp_base": (38.0, 58.0),
        "vib_base": (8.0, 20.0),
        "current_base": (8.0, 19.0),
        "rpm_base": (15.0, 45.0),
        "kt_ratio": 5.0,  # Torque ~ kt * current
    },
    "elbow": {
        "torque_base": (25.0, 70.0),
        "temp_base": (35.0, 55.0),
        "vib_base": (6.0, 18.0),
        "current_base": (5.0, 14.0),
        "rpm_base": (20.0, 60.0),
        "kt_ratio": 5.0,
    },
    "wrist": {
        "torque_base": (8.0, 28.0),
        "temp_base": (30.0, 50.0),
        "vib_base": (4.0, 15.0),
        "current_base": (1.8, 6.0),
        "rpm_base": (30.0, 90.0),
        "kt_ratio": 4.5,
    },
}

def generate_sample(sample_id: int, arm_id: str, joint: str, wear_state: str) -> dict:
    spec = JOINT_SPECS[joint]
    kt = spec["kt_ratio"]

    if wear_state == "NORMAL":
        torque = random.uniform(*spec["torque_base"])
        temp = random.uniform(*spec["temp_base"])
        vib = random.uniform(*spec["vib_base"])
        current = (torque / kt) * random.uniform(0.95, 1.05)
        operating_hours = random.randint(50, 800)

    elif wear_state == "EARLY_WEAR":
        # Slight elevation in vibration and operating temperature
        torque = random.uniform(*spec["torque_base"]) * random.uniform(1.05, 1.15)
        temp = random.uniform(spec["temp_base"][1], spec["temp_base"][1] + 10.0)
        vib = random.uniform(spec["vib_base"][1], spec["vib_base"][1] + 8.0)
        current = (torque / kt) * random.uniform(0.98, 1.10)
        operating_hours = random.randint(800, 2500)

    elif wear_state == "ACCELERATED_WEAR":
        # Clear bearing degradation, higher friction torque and vibration
        torque = random.uniform(*spec["torque_base"]) * random.uniform(1.20, 1.45)
        temp = random.uniform(spec["temp_base"][1] + 8.0, spec["temp_base"][1] + 20.0)
        vib = random.uniform(spec["vib_base"][1] + 8.0, spec["vib_base"][1] + 22.0)
        current = (torque / kt) * random.uniform(1.05, 1.25)
        operating_hours = random.randint(2500, 5000)

    elif wear_state == "CRITICAL_WEAR":
        # Severe friction, high thermal accumulation, dangerous harmonic vibration
        torque = random.uniform(*spec["torque_base"]) * random.uniform(1.45, 1.85)
        temp = random.uniform(spec["temp_base"][1] + 20.0, spec["temp_base"][1] + 35.0)
        vib = random.uniform(spec["vib_base"][1] + 22.0, spec["vib_base"][1] + 40.0)
        current = (torque / kt) * random.uniform(1.20, 1.50)
        operating_hours = random.randint(5000, 9000)

    else:
        raise ValueError(f"Unknown wear state: {wear_state}")

    speed_rpm = random.uniform(*spec["rpm_base"])

    return {
        "sample_id": f"REC_{sample_id:04d}",
        "arm_id": arm_id,
        "joint": joint,
        "torque_nm": round(torque, 2),
        "temperature_c": round(temp, 2),
        "vibration_mms": round(vib, 2),
        "motor_current_a": round(current, 2),
        "speed_rpm": round(speed_rpm, 1),
        "operating_hours": operating_hours,
        "wear_state": wear_state,
    }

def generate_dataset(total_records: int = 150) -> list:
    random.seed(42)
    records = []
    arms = [f"Arm_{i}" for i in range(1, 9)]
    joints = ["shoulder", "elbow", "wrist"]
    states = ["NORMAL", "EARLY_WEAR", "ACCELERATED_WEAR", "CRITICAL_WEAR"]
    weights = [0.45, 0.25, 0.18, 0.12]

    for idx in range(1, total_records + 1):
        arm = random.choice(arms)
        joint = random.choice(joints)
        wear = random.choices(states, weights=weights, k=1)[0]
        rec = generate_sample(idx, arm, joint, wear)
        records.append(rec)

    return records

def save_csv(records: list, file_path: Path):
    fieldnames = [
        "sample_id",
        "arm_id",
        "joint",
        "torque_nm",
        "temperature_c",
        "vibration_mms",
        "motor_current_a",
        "speed_rpm",
        "operating_hours",
        "wear_state",
    ]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Dataset successfully created with {len(records)} records at {file_path}")

if __name__ == "__main__":
    recs = generate_dataset(150)
    save_csv(recs, CSV_PATH)
