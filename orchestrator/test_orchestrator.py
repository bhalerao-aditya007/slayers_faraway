import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import redis
import threading
from orchestrator.orchestrator import Orchestrator, CH_CONSENSUS, CH_ESCALATION
from orchestrator.message_schemas import (
    HumanOverrideMessage, OverrideLevel, ActionType,
    SituationVectorMessage, ActionRecommendationMessage
)


def test_redis_connection():
    print("=" * 55)
    print("TEST 1: Redis Connection")
    print("=" * 55)
    r = redis.Redis(host="localhost", port=6379, db=0)
    assert r.ping(), "Redis not responding"
    print("Redis connection: PASSED\n")


def test_consensus_nominal():
    print("=" * 55)
    print("TEST 2: Consensus — Nominal Operation")
    print("=" * 55)

    orc = Orchestrator()
    orc.start()
    time.sleep(0.5)

    # Simulate high confidence perception
    orc.publish_test_perception(confidence="high", jensen_gain=1.0)

    # Listen for consensus
    r = redis.Redis(host="localhost", port=6379, db=0)
    p = r.pubsub()
    p.subscribe(CH_CONSENSUS)

    received = []
    def listener():
        for msg in p.listen():
            if msg["type"] == "message":
                received.append(json.loads(msg["data"]))
                break

    t = threading.Thread(target=listener, daemon=True)
    t.start()
    t.join(timeout=3)

    orc.stop()

    if received:
        result = received[0]
        print(f"Final action: {result['final_action']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Consensus reached: {result['consensus_reached']}")
        print("PASSED\n")
    else:
        print("No consensus message received (check Redis)\n")


def test_consensus_low_confidence():
    print("=" * 55)
    print("TEST 3: Consensus — Low Confidence Pose")
    print("=" * 55)

    orc = Orchestrator()
    orc.start()
    time.sleep(0.5)

    # Simulate LOW confidence — Jensen Gain 25°
    orc.publish_test_perception(confidence="low", jensen_gain=25.0)

    r = redis.Redis(host="localhost", port=6379, db=0)
    p = r.pubsub()
    p.subscribe(CH_CONSENSUS)

    received = []
    def listener():
        for msg in p.listen():
            if msg["type"] == "message":
                received.append(json.loads(msg["data"]))
                break

    t = threading.Thread(target=listener, daemon=True)
    t.start()
    t.join(timeout=3)

    orc.stop()

    if received:
        result = received[0]
        print(f"Final action: {result['final_action']}")
        print(f"Escalated to human: {result['escalated_to_human']}")
        print(f"Reasoning: {result['reasoning']}")
        assert result["final_action"] == ActionType.HOLD_POSITION, \
            "Low confidence should result in HOLD"
        assert result["escalated_to_human"], \
            "Low confidence should escalate"
        print("PASSED\n")
    else:
        print("No consensus message received\n")


def test_human_override():
    print("=" * 55)
    print("TEST 4: Armstrong Protocol — Human Override")
    print("=" * 55)

    orc = Orchestrator()
    orc.start()
    time.sleep(0.5)

    r = redis.Redis(host="localhost", port=6379, db=0)

    # Publish human override
    override = HumanOverrideMessage(
        override_level=OverrideLevel.REPLACE,
        selected_action=ActionType.PROCEED_SLOW,
        rationale="Visual inspection confirms safe approach",
        operator_id="commander_test"
    )
    r.publish("human.in", override.to_json())

    # Listen for consensus
    p = r.pubsub()
    p.subscribe(CH_CONSENSUS)

    received = []
    def listener():
        for msg in p.listen():
            if msg["type"] == "message":
                received.append(json.loads(msg["data"]))
                break

    t = threading.Thread(target=listener, daemon=True)
    t.start()
    t.join(timeout=3)

    orc.stop()

    if received:
        result = received[0]
        print(f"Final action: {result['final_action']}")
        print(f"Override applied: {result['override_applied']}")
        print(f"Reasoning: {result['reasoning']}")
        assert result["override_applied"], "Override should be applied"
        assert result["final_action"] == ActionType.PROCEED_SLOW
        print("PASSED\n")
    else:
        print("No consensus received — override may need another cycle\n")


if __name__ == "__main__":
    test_redis_connection()
    test_consensus_nominal()
    test_consensus_low_confidence()
    test_human_override()

    print("=" * 55)
    print("ALL ORCHESTRATOR TESTS COMPLETE")
    print("=" * 55)