import json
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from .message_schemas import (
    ConfidenceLevel, ActionType,
    PoseEstimateMessage, SituationVectorMessage,
    ActionRecommendationMessage, HumanOverrideMessage
)


@dataclass
class SharedState:
    """
    Single source of truth for all agents.
    All agents READ from this. Only Orchestrator WRITES to this.
    """

    # Latest perception output
    latest_pose:            Optional[dict] = None
    pose_timestamp:         float = 0.0
    pose_confidence:        str   = ConfidenceLevel.HIGH
    pose_trustworthy:       bool  = True
    jensen_gain:            float = 0.0

    # Latest cognition output
    latest_situation:       Optional[dict] = None
    situation_timestamp:    float = 0.0
    anomaly_detected:       bool  = False
    anomaly_severity:       str   = "none"
    novelty_score:          float = 0.0

    # Latest action recommendation
    latest_recommendation:  Optional[dict] = None
    recommendation_timestamp: float = 0.0
    recommended_action:     str   = ActionType.HOLD_POSITION

    # Human override state
    last_override:          Optional[dict] = None
    override_timestamp:     float = 0.0
    manual_control_active:  bool  = False

    # Mission state
    mission_phase:          str   = "approach"
    mission_elapsed_s:      float = 0.0
    mission_start_time:     float = field(default_factory=time.time)

    # System health
    agent_last_heartbeat:   dict  = field(default_factory=dict)
    agent_alive:            dict  = field(default_factory=dict)

    # Decision history (last 100 decisions)
    decision_history:       list  = field(default_factory=list)
    total_decisions:        int   = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def is_stale(self, timestamp: float, max_age_s: float = 5.0) -> bool:
        """Check if a timestamp is too old to be trusted."""
        return (time.time() - timestamp) > max_age_s

    def pose_is_stale(self) -> bool:
        return self.is_stale(self.pose_timestamp, max_age_s=2.0)

    def situation_is_stale(self) -> bool:
        return self.is_stale(self.situation_timestamp, max_age_s=5.0)


class StateManager:
    """
    Thread-safe shared state for all agents.
    Uses a read-write lock pattern.
    """

    def __init__(self):
        self._state = SharedState()
        self._lock = threading.RLock()
        self._history_limit = 100

    def get_state(self) -> SharedState:
        with self._lock:
            return self._state

    def update_from_perception(self, msg: PoseEstimateMessage):
        with self._lock:
            self._state.latest_pose = {
                "R": msg.R,
                "t": msg.t,
                "quaternion": msg.quaternion
            }
            self._state.pose_timestamp         = msg.timestamp
            self._state.pose_confidence        = msg.confidence_level
            self._state.pose_trustworthy       = msg.is_trustworthy
            self._state.jensen_gain            = msg.jensen_gain

    def update_from_cognition(self, msg: SituationVectorMessage):
        with self._lock:
            self._state.latest_situation       = asdict(msg)
            self._state.situation_timestamp    = msg.timestamp
            self._state.anomaly_detected       = msg.anomaly_detected
            self._state.anomaly_severity       = msg.anomaly_severity
            self._state.novelty_score          = msg.novelty_score
            self._state.recommended_action     = msg.recommended_action

    def update_from_action(self, msg: ActionRecommendationMessage):
        with self._lock:
            self._state.latest_recommendation  = asdict(msg)
            self._state.recommendation_timestamp = msg.timestamp
            self._state.recommended_action     = msg.primary_action

    def update_from_human(self, msg: HumanOverrideMessage):
        with self._lock:
            self._state.last_override          = asdict(msg)
            self._state.override_timestamp     = msg.timestamp
            from orchestrator.message_schemas import OverrideLevel
            self._state.manual_control_active  = (
                msg.override_level == OverrideLevel.REJECT
            )

    def record_decision(self, action: str, reasoning: str,
                        consensus: bool, override: bool):
        with self._lock:
            entry = {
                "timestamp":  time.time(),
                "action":     action,
                "reasoning":  reasoning,
                "consensus":  consensus,
                "override":   override,
                "total_num":  self._state.total_decisions
            }
            self._state.decision_history.append(entry)
            if len(self._state.decision_history) > self._history_limit:
                self._state.decision_history.pop(0)
            self._state.total_decisions += 1

    def update_agent_heartbeat(self, agent_id: str):
        with self._lock:
            self._state.agent_last_heartbeat[agent_id] = time.time()
            self._state.agent_alive[agent_id] = True

    def check_agent_health(self, timeout_s: float = 10.0):
        with self._lock:
            now = time.time()
            for agent_id, last_hb in self._state.agent_last_heartbeat.items():
                self._state.agent_alive[agent_id] = (now - last_hb) < timeout_s