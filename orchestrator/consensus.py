from typing import Dict, Optional
from .message_schemas import (
    ActionType, ConfidenceLevel, OverrideLevel,
    PoseEstimateMessage, SituationVectorMessage,
    ActionRecommendationMessage, HumanOverrideMessage,
    ConsensusActionMessage
)
from .state_manager import SharedState


# Conservative action ordering — lower index = safer
SAFETY_RANKING = [
    ActionType.ABORT,
    ActionType.EMERGENCY_VENT,
    ActionType.HOLD_POSITION,
    ActionType.ISOLATE_MODULE,
    ActionType.RECONFIGURE_POWER,
    ActionType.PROCEED_SLOW,
    ActionType.PROCEED_NORMAL,
    ActionType.AWAIT_HUMAN,
    ActionType.AUTONOMOUS_FALLBACK,
]


def most_conservative(actions: list) -> str:
    """Return the most conservative action from a list."""
    ranked = []
    for a in actions:
        try:
            ranked.append((SAFETY_RANKING.index(a), a))
        except ValueError:
            ranked.append((len(SAFETY_RANKING), a))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1] if ranked else ActionType.HOLD_POSITION


class ConsensusEngine:
    """
    Implements the consensus and conflict resolution protocol.

    Rules (from outline Step 5.3):
    1. If Perception confidence LOW  -> Cognition must use conservative policy
    2. If Cognition flags novelty    -> Action defaults to HOLD
    3. If Action proposes high-risk  -> require human confirmation
    4. Tie-breaker: most conservative action wins
    5. Any CRITICAL uncertainty      -> escalate to human
    6. Human override always wins    -> apply immediately
    """

    # Weights for voting (tunable)
    AGENT_WEIGHTS = {
        "perception": 0.3,
        "cognition":  0.4,
        "action":     0.3,
    }

    # High-risk actions that always need human confirmation
    HIGH_RISK_ACTIONS = {
        ActionType.EMERGENCY_VENT,
        ActionType.ABORT,
    }

    def run(self,
            state: SharedState,
            perception_msg: Optional[PoseEstimateMessage] = None,
            cognition_msg: Optional[SituationVectorMessage] = None,
            action_msg: Optional[ActionRecommendationMessage] = None,
            human_msg: Optional[HumanOverrideMessage] = None
            ) -> ConsensusActionMessage:
        """
        Core consensus logic. Called every decision cycle.

        Returns a ConsensusActionMessage with the final decided action.
        """
        votes = {}
        reasoning_parts = []
        escalate = False
        fallback = False

        # ── Rule 0: Human override always wins ──────────────────────────
        if human_msg is not None:
            return self._apply_human_override(human_msg, votes)

        # ── Rule 1: Perception confidence check ─────────────────────────
        perception_vote = ActionType.HOLD_POSITION
        if perception_msg is not None:
            if not perception_msg.is_trustworthy:
                # Force conservative — low confidence pose
                perception_vote = ActionType.HOLD_POSITION
                reasoning_parts.append(
                    f"Perception LOW confidence "
                    f"(Jensen Gain {perception_msg.jensen_gain:.1f}°) "
                    f"-> forcing HOLD"
                )
                escalate = True
            else:
                perception_vote = ActionType.PROCEED_SLOW
                reasoning_parts.append(
                    f"Perception OK "
                    f"(Jensen Gain {perception_msg.jensen_gain:.1f}°)"
                )
        else:
            # No perception data -> stale -> conservative
            perception_vote = ActionType.HOLD_POSITION
            reasoning_parts.append("No perception data -> HOLD")
            fallback = True

        votes["perception"] = perception_vote

        # ── Rule 2: Cognition novelty check ─────────────────────────────
        cognition_vote = ActionType.HOLD_POSITION
        if cognition_msg is not None:
            if cognition_msg.novelty_score > 0.7:
                cognition_vote = ActionType.HOLD_POSITION
                reasoning_parts.append(
                    f"Cognition NOVEL situation "
                    f"(score {cognition_msg.novelty_score:.2f}) -> HOLD"
                )
                escalate = True
            elif cognition_msg.anomaly_detected:
                cognition_vote = cognition_msg.recommended_action
                reasoning_parts.append(
                    f"Cognition: anomaly {cognition_msg.anomaly_type} "
                    f"({cognition_msg.anomaly_severity}) "
                    f"-> {cognition_msg.recommended_action}"
                )
            else:
                cognition_vote = cognition_msg.recommended_action
                reasoning_parts.append(
                    f"Cognition: nominal -> {cognition_msg.recommended_action}"
                )
        else:
            cognition_vote = ActionType.HOLD_POSITION
            reasoning_parts.append("No cognition data -> HOLD")
            fallback = True

        votes["cognition"] = cognition_vote

        # ── Rule 3: Action agent high-risk check ────────────────────────
        action_vote = ActionType.HOLD_POSITION
        if action_msg is not None:
            if action_msg.primary_action in self.HIGH_RISK_ACTIONS:
                # High risk — don't vote for it, escalate to human
                action_vote = ActionType.AWAIT_HUMAN
                reasoning_parts.append(
                    f"Action proposed HIGH-RISK {action_msg.primary_action} "
                    f"-> escalating to human"
                )
                escalate = True
            else:
                action_vote = action_msg.primary_action
                reasoning_parts.append(
                    f"Action: {action_msg.primary_action} "
                    f"(collision_prob={action_msg.collision_prob:.2f})"
                )
        else:
            action_vote = ActionType.HOLD_POSITION
            reasoning_parts.append("No action data -> HOLD")
            fallback = True

        votes["action"] = action_vote

        # ── Weighted voting ──────────────────────────────────────────────
        all_votes = list(votes.values())
        unique_actions = set(all_votes)

        if len(unique_actions) == 1:
            # Full consensus
            final_action = all_votes[0]
            consensus_reached = True
            reasoning_parts.append("Full consensus reached")
        else:
            # Conflict -> most conservative wins (from outline)
            final_action = most_conservative(all_votes)
            consensus_reached = False
            reasoning_parts.append(
                f"Conflict {[v for v in all_votes]} "
                f"-> conservative tiebreak: {final_action}"
            )
            if not escalate:
                escalate = True  # Always escalate on conflict

        return ConsensusActionMessage(
            final_action=final_action,
            consensus_reached=consensus_reached,
            votes=votes,
            override_applied=False,
            override_level="",
            escalated_to_human=escalate,
            reasoning=" | ".join(reasoning_parts),
            fallback_triggered=fallback
        )

    def _apply_human_override(self,
                              human_msg: HumanOverrideMessage,
                              votes: dict) -> ConsensusActionMessage:
        """Human override immediately becomes final action."""
        return ConsensusActionMessage(
            final_action=human_msg.selected_action,
            consensus_reached=True,
            votes=votes,
            override_applied=True,
            override_level=human_msg.override_level,
            escalated_to_human=False,
            reasoning=f"HUMAN OVERRIDE Level {human_msg.override_level}: "
                      f"{human_msg.selected_action} | "
                      f"Rationale: {human_msg.rationale}",
            fallback_triggered=False
        )