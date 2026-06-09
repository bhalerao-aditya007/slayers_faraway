import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import threading
import redis

from simulation.scenario_engine import ScenarioEngine
from simulation.scenarios.scenario_library import (
    nominal_docking,
    thermal_anomaly,
    perception_challenge,
    perfect_storm
)
from orchestrator.orchestrator import Orchestrator


def run_scenario_with_orchestrator(scenario, speed: float = 5.0):
    """
    Run a scenario with the full orchestrator active.
    speed=5.0 means 5x faster than real time.
    """
    print(f"\n{'='*55}")
    print(f"RUNNING: {scenario.name}")
    print(f"{'='*55}")

    # Start orchestrator
    orc = Orchestrator()
    orc.start()
    time.sleep(0.5)

    # Track consensus decisions
    decisions = []
    escalations = []

    r = redis.Redis(host="localhost", port=6379, db=0)
    p = r.pubsub()
    p.subscribe("orchestrator.consensus", "orchestrator.escalation")

    def listen():
        for msg in p.listen():
            if msg["type"] != "message":
                continue
            channel = msg["channel"].decode()
            data = json.loads(msg["data"])
            if channel == "orchestrator.consensus":
                decisions.append(data)
            elif channel == "orchestrator.escalation":
                escalations.append(data)

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()

    # Run scenario
    engine = ScenarioEngine()
    engine.run_scenario(scenario, speed=speed)

    # Wait for final decisions
    time.sleep(1.0)
    orc.stop()

    # Print results
    print(f"\nRESULTS for '{scenario.name}':")
    print(f"  Total decisions: {len(decisions)}")
    print(f"  Total escalations: {len(escalations)}")

    if decisions:
        actions = [d["final_action"] for d in decisions]
        unique = set(actions)
        print(f"  Actions taken: {unique}")
        overrides = sum(1 for d in decisions if d["override_applied"])
        print(f"  Human overrides: {overrides}")
        consensus = sum(1 for d in decisions if d["consensus_reached"])
        print(f"  Consensus rate: {consensus}/{len(decisions)}")

    return decisions, escalations


if __name__ == "__main__":
    print("PHASE 6 — SIMULATION ENVIRONMENT TEST")
    print("Running all scenarios at 5x speed\n")

    # Test 1: Nominal
    d1, e1 = run_scenario_with_orchestrator(nominal_docking(), speed=5.0)
    print(f"Nominal: {len(e1)} escalations (expected: 0)\n")

    # Test 2: Thermal anomaly
    d2, e2 = run_scenario_with_orchestrator(thermal_anomaly(), speed=5.0)
    print(f"Thermal: {len(e2)} escalations (expected: >0)\n")

    # Test 3: Perception challenge
    d3, e3 = run_scenario_with_orchestrator(perception_challenge(), speed=5.0)
    print(f"Perception: {len(e3)} escalations (expected: >0)\n")

    # Test 4: Perfect storm
    d4, e4 = run_scenario_with_orchestrator(perfect_storm(), speed=10.0)
    print(f"Perfect Storm: {len(e4)} escalations (expected: many)\n")

    print("="*55)
    print("ALL SIMULATION TESTS COMPLETE")
    print("="*55)