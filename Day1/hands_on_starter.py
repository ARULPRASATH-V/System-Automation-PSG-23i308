import random
import time

# The 5 factory machines we are tracking
MACHINES = ["Machine_A", "Machine_B", "Machine_C", "Machine_D", "Machine_E"]

# Alert rules - lower priority numbers are checked first
RULES = [
    {"name": "CRITICAL_OVERLOAD", "priority": 1, "condition": lambda r: r["current"] is not None and r["current"] > 28, "action": "CUT_POWER"},
    {"name": "HIGH_TEMP",         "priority": 2, "condition": lambda r: r["temperature"] is not None and r["temperature"] > 85, "action": "STOP_MACHINE"},
    {"name": "HIGH_PRESSURE",     "priority": 3, "condition": lambda r: r["pressure"] is not None and r["pressure"] > 45, "action": "VENT_PRESSURE"},
]

def generate_factory_data():
    """Simulates live data readings from our sensors."""
    data_map = {}
    for name in MACHINES:
        # 5% chance of simulating a system hazard/fault
        broken = random.random() < 0.05
        data_map[name] = {
            "temperature": random.uniform(90, 100) if broken else random.uniform(50, 80),
            "pressure":    random.uniform(46, 55)  if broken else random.uniform(20, 40),
            "current":     random.uniform(29, 35)  if broken else random.uniform(5, 20)
        }
    return data_map

def analyze_factory(factory_data):
    """Scans all machines to find which ones are breaking rules."""
    found_issues = []
   
    for machine_id, current_reading in factory_data.items():
        # Look through rules starting at priority 1
        for rule in sorted(RULES, key=lambda x: x["priority"]):
            if rule["condition"](current_reading):
                found_issues.append({
                    "machine": machine_id,
                    "rule_name": rule["name"],
                    "priority": rule["priority"],
                    "action": rule["action"]
                })
                break # Stop checking this machine if a higher rule already broke
               
    # Sort everything by urgency before returning
    found_issues.sort(key=lambda x: x["priority"])
    return found_issues

def main():
    print("--- SYSTEM SCANNER RUNNING ---")
    print("Reading current telemetry stats...\n")
   
    # Grab data and run it through the engine
    data = generate_factory_data()
    bugs = analyze_factory(data)
   
    # If the list is not empty, handle the problems
    if len(bugs) > 0:
        worst = bugs[0] # Top item is the absolute worst emergency
        print(f"!!! FIX THIS IMMEDIATELY: {worst['machine']} !!!")
        print(f"Danger Type: {worst['rule_name']}")
        print(f"Required Fix: {worst['action']}\n")
       
        print("All active system warnings:")
        for alert in bugs:
            print(f" -> {alert['machine']} | Priority {alert['priority']} | Rule: {alert['rule_name']}")
    else:
        print("Status Check: Everything looks good. No active alerts.")

if __name__ == "__main__":
    main()