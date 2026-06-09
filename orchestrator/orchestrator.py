"""
Orchestrator Agent — coordinates all sub-agents via Redis pub/sub.
"""

import json
import time
import threading
import redis
from typing import Optional

from .message_schemas import (
    AgentID, MessageType,
    PoseEstimateMessage, SituationVectorMessage,
    ActionRecommendationMessage, HumanOverrideMessage,
    ConsensusActionMessage, EscalationMessage,
    SystemStatusMessage, ConfidenceLevel
)
from .state_manager import StateManager
from .consensus import ConsensusEngine
from .armstrong_protocol import ArmstrongProtocol


# Redis channel names
CH_PERCEPTION   = "perception.out"
CH_COGNITION    = "cognition.out"
CH_ACTION       = "action.out"
CH_INTERFACE    = "interface.out"
CH_HUMAN_IN     = "human.in"
CH_CONSENSUS    = "orchestrator.consensus"
CH_ESCALATION   = "orchestrator.escalation"
CH_STATUS       = "orchestrator.status"
CH_HEARTBEAT    = "orchestrator.heartbeat"

ALL_CHANNELS = [
    CH_PERCEPTION, CH_COGNITION, CH_ACTION,
    CH_INTERFACE, CH_HUMAN_IN
]


class Orchestrator:
    """
    Central coordinator for the spacecraft autonomy system.

    Subscribes to all agent output channels.
    Runs consensus every decision cycle.
    Publishes final action to all agents.
    Handles escalation via Armstrong Protocol.
    """

    DECISION_CYCLE_S = 1.0   # Run consensus every 1 second
    STATUS_CYCLE_S   = 5.0   # Broadcast system status every 5 seconds

    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 decision_timeout_s: int = 30):
        self.redis_host = redis_host
        self.redis_port = redis_port

        # Redis connections
        self.redis_pub  = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.redis_sub  = redis.Redis(host=redis_host, port=redis_port, db=0)

        # Core components
        self.state      = StateManager()
        self.consensus  = ConsensusEngine()
        self.armstrong  = ArmstrongProtocol(
            timeout_s=decision_timeout_s,
            on_timeout=self._on_armstrong_timeout,
            on_override=self._on_human_override
        )

        # Latest messages from each agent
        self._latest_perception: Optional[PoseEstimateMessage] = None
        self._latest_cognition:  Optional[SituationVectorMessage] = None
        self._latest_action:     Optional[ActionRecommendationMessage] = None
        self._latest_human:      Optional[HumanOverrideMessage] = None
        self._msg_lock = threading.Lock()

        # Control flags
        self._running   = False
        self._cycle_count = 0

        print(f"Orchestrator initialized")
        print(f"  Redis: {redis_host}:{redis_port}")
        print(f"  Decision cycle: {self.DECISION_CYCLE_S}s")
        print(f"  Armstrong timeout: {decision_timeout_s}s")

    def start(self):
        """Start all orchestrator threads."""
        self._running = True

        # Subscriber thread
        self._sub_thread = threading.Thread(
            target=self._subscriber_loop,
            daemon=True,
            name="orchestrator-subscriber"
        )
        self._sub_thread.start()

        # Decision cycle thread
        self._decision_thread = threading.Thread(
            target=self._decision_loop,
            daemon=True,
            name="orchestrator-decision"
        )
        self._decision_thread.start()

        # Status broadcast thread
        self._status_thread = threading.Thread(
            target=self._status_loop,
            daemon=True,
            name="orchestrator-status"
        )
        self._status_thread.start()

        print("Orchestrator started — all threads running")

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        print("Orchestrator stopped")

    def _subscriber_loop(self):
        """
        Listens to all agent channels.
        Parses messages and updates latest state.
        """
        pubsub = self.redis_sub.pubsub()
        pubsub.subscribe(*ALL_CHANNELS)

        print(f"Subscribed to channels: {ALL_CHANNELS}")

        for raw_msg in pubsub.listen():
            if not self._running:
                break
            if raw_msg["type"] != "message":
                continue

            channel = raw_msg["channel"].decode()
            data    = raw_msg["data"].decode()

            try:
                self._route_message(channel, data)
            except Exception as e:
                print(f"[Orchestrator] Error routing message "
                      f"from {channel}: {e}")

    def _route_message(self, channel: str, data: str):
        """Parse and route incoming message to correct handler."""
        payload = json.loads(data)

        with self._msg_lock:
            if channel == CH_PERCEPTION:
                self._latest_perception = PoseEstimateMessage(**payload)
                self.state.update_from_perception(self._latest_perception)
                print(f"[Orchestrator] Perception: "
                      f"confidence={self._latest_perception.confidence_level} "
                      f"JG={self._latest_perception.jensen_gain:.1f}°")

            elif channel == CH_COGNITION:
                self._latest_cognition = SituationVectorMessage(**payload)
                self.state.update_from_cognition(self._latest_cognition)
                print(f"[Orchestrator] Cognition: "
                      f"anomaly={self._latest_cognition.anomaly_detected} "
                      f"novelty={self._latest_cognition.novelty_score:.2f}")

            elif channel == CH_ACTION:
                self._latest_action = ActionRecommendationMessage(**payload)
                self.state.update_from_action(self._latest_action)
                print(f"[Orchestrator] Action: "
                      f"{self._latest_action.primary_action}")

            elif channel == CH_HUMAN_IN:
                self._latest_human = HumanOverrideMessage(**payload)
                self.state.update_from_human(self._latest_human)
                self.armstrong.receive_override(self._latest_human)
                print(f"[Orchestrator] Human override: "
                      f"Level {self._latest_human.override_level} "
                      f"-> {self._latest_human.selected_action}")

    def _decision_loop(self):
        """
        Runs consensus every DECISION_CYCLE_S seconds.
        Publishes result to orchestrator.consensus channel.
        """
        while self._running:
            cycle_start = time.time()

            with self._msg_lock:
                p = self._latest_perception
                c = self._latest_cognition
                a = self._latest_action
                h = self._latest_human
                # Consume human override (one-shot)
                self._latest_human = None

            # Run consensus
            result = self.consensus.run(
                state=self.state.get_state(),
                perception_msg=p,
                cognition_msg=c,
                action_msg=a,
                human_msg=h
            )

            # Record decision
            self.state.record_decision(
                action=result.final_action,
                reasoning=result.reasoning,
                consensus=result.consensus_reached,
                override=result.override_applied
            )

            # Publish consensus to all agents
            self._publish(CH_CONSENSUS, result.to_json())

            # Escalate to human if needed
            if result.escalated_to_human and not result.override_applied:
                self._escalate(result)

            self._cycle_count += 1

            # Maintain cycle timing
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.DECISION_CYCLE_S - elapsed)
            time.sleep(sleep_time)

    def _escalate(self, result: ConsensusActionMessage):
        """Publish escalation notice to interface agent."""
        esc = EscalationMessage(
            reason=result.reasoning,
            urgency=ConfidenceLevel.MODERATE,
            suggested_action=result.final_action,
            timeout_seconds=self.armstrong.timeout_s,
            context={
                "consensus_reached": result.consensus_reached,
                "votes": result.votes,
                "fallback": result.fallback_triggered
            }
        )
        self._publish(CH_ESCALATION, esc.to_json())

    def _status_loop(self):
        """Broadcast system health every STATUS_CYCLE_S seconds."""
        while self._running:
            time.sleep(self.STATUS_CYCLE_S)
            self.state.check_agent_health(timeout_s=10.0)
            s = self.state.get_state()

            status = SystemStatusMessage(
                perception_alive=s.agent_alive.get("perception", False),
                cognition_alive=s.agent_alive.get("cognition", False),
                action_alive=s.agent_alive.get("action", False),
                interface_alive=s.agent_alive.get("interface", False),
                overall_status="nominal" if all([
                    s.agent_alive.get("perception", False),
                    s.agent_alive.get("cognition", False),
                ]) else "degraded",
                cycle_time_ms=self.DECISION_CYCLE_S * 1000,
                total_cycles=self._cycle_count
            )
            self._publish(CH_STATUS, status.to_json())

    def _publish(self, channel: str, message: str):
        """Publish message to Redis channel."""
        try:
            self.redis_pub.publish(channel, message)
        except Exception as e:
            print(f"[Orchestrator] Publish error on {channel}: {e}")

    def _on_armstrong_timeout(self):
        """Called when Armstrong Protocol times out."""
        print("[Orchestrator] Armstrong timeout — publishing HOLD_POSITION")

    def _on_human_override(self, msg: HumanOverrideMessage):
        """Called when human override received."""
        print(f"[Orchestrator] Override logged: {msg.override_level}")

    def publish_test_perception(self, confidence: str = "high",
                                jensen_gain: float = 1.0):
        """Helper for testing — simulate a perception message."""
        from orchestrator.message_schemas import ActionType
        msg = PoseEstimateMessage(
            R=[[1,0,0],[0,1,0],[0,0,1]],
            t=[0.0, 0.0, 10.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            jensen_gain=jensen_gain,
            confidence_level=confidence,
            confidence_label=f"{confidence.upper()} CONFIDENCE",
            sigma_R_deg=jensen_gain * 0.6,
            sigma_t_m=0.1,
            nearest_anchor_idx=0,
            anchor_distance_deg=5.0,
            is_trustworthy=confidence in ("high", "moderate"),
            processing_time_ms=33.0,
            image_shape=[64, 64, 3]
        )
        self._publish(CH_PERCEPTION, msg.to_json())
        return msg