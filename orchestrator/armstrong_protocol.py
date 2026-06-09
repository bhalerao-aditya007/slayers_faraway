"""
Armstrong Protocol — Structured Override Handling

Named after Neil Armstrong's manual override of the Apollo 11
landing computer. The protocol ensures humans can always
override AI decisions with full logging and timeout safety.
"""

import time
import threading
from typing import Optional, Callable
from .message_schemas import (
    OverrideLevel, ActionType, HumanOverrideMessage
)


class ArmstrongProtocol:
    """
    Manages structured human override with timeout fallback.

    Override Levels:
        Level 1 ACKNOWLEDGE — Accept AI recommendation, proceed
        Level 2 MODIFY      — Adjust parameters
        Level 3 REPLACE     — Select different action
        Level 4 REJECT      — Full manual control

    Timeout behavior:
        If human doesn't respond within timeout_s:
        -> Default to most conservative action
        -> Log as "autonomous fallback"
    """

    def __init__(self,
                 timeout_s: int = 30,
                 on_timeout: Optional[Callable] = None,
                 on_override: Optional[Callable] = None):
        """
        Args:
            timeout_s:   seconds to wait for human response
            on_timeout:  callback when timeout fires
            on_override: callback when override received
        """
        self.timeout_s   = timeout_s
        self.on_timeout  = on_timeout
        self.on_override = on_override

        self._waiting_for_human = False
        self._override_event    = threading.Event()
        self._latest_override: Optional[HumanOverrideMessage] = None
        self._timeout_timer: Optional[threading.Timer] = None

        # Full override history for learning
        self.override_history: list = []

    def request_human_input(self,
                            suggested_action: str,
                            reason: str,
                            urgency: str = "moderate",
                            timeout_s: Optional[int] = None) -> dict:
        """
        Request human input. Blocks until response or timeout.

        Args:
            suggested_action: what AI wants to do
            reason:           why human input is needed
            urgency:          "low" / "moderate" / "critical"
            timeout_s:        override instance timeout

        Returns:
            dict with keys: action, override_level, rationale, timed_out
        """
        t = timeout_s or self.timeout_s
        self._waiting_for_human = True
        self._override_event.clear()
        self._latest_override = None

        print(f"\n[ARMSTRONG PROTOCOL] Human input requested")
        print(f"  Urgency:   {urgency.upper()}")
        print(f"  Reason:    {reason}")
        print(f"  Suggested: {suggested_action}")
        print(f"  Timeout:   {t}s")

        # Start timeout timer
        self._timeout_timer = threading.Timer(t, self._handle_timeout)
        self._timeout_timer.start()

        # Wait for human response or timeout
        responded = self._override_event.wait(timeout=t + 1)

        self._waiting_for_human = False

        if not responded or self._latest_override is None:
            return self._timeout_response(suggested_action)
        else:
            return self._override_response(self._latest_override)

    def receive_override(self, msg: HumanOverrideMessage):
        """
        Called when human sends an override command.
        Thread-safe — can be called from Redis subscriber thread.
        """
        if not self._waiting_for_human:
            # Unsolicited override — still valid, log it
            print(f"[ARMSTRONG] Unsolicited override received: "
                  f"Level {msg.override_level} -> {msg.selected_action}")

        # Cancel timeout timer
        if self._timeout_timer:
            self._timeout_timer.cancel()

        self._latest_override = msg
        self._override_event.set()

        # Log to history
        self.override_history.append({
            "timestamp":     time.time(),
            "override_level": msg.override_level,
            "action":        msg.selected_action,
            "rationale":     msg.rationale,
            "timed_out":     False
        })

        if self.on_override:
            self.on_override(msg)

    def _handle_timeout(self):
        """Called by timer thread when human doesn't respond."""
        print(f"\n[ARMSTRONG] TIMEOUT — no human response in {self.timeout_s}s")
        print(f"[ARMSTRONG] Defaulting to most conservative action")
        self._override_event.set()  # Unblock the wait

        if self.on_timeout:
            self.on_timeout()

    def _timeout_response(self, suggested_action: str) -> dict:
        """Build response for timeout case."""
        # On timeout always go to HOLD (safest)
        fallback = ActionType.HOLD_POSITION

        self.override_history.append({
            "timestamp":      time.time(),
            "override_level": "autonomous_fallback",
            "action":         fallback,
            "rationale":      "timeout",
            "timed_out":      True
        })

        print(f"[ARMSTRONG] Autonomous fallback -> {fallback}")
        return {
            "action":         fallback,
            "override_level": "autonomous_fallback",
            "rationale":      f"Human timeout after {self.timeout_s}s",
            "timed_out":      True
        }

    def _override_response(self, msg: HumanOverrideMessage) -> dict:
        """Build response from human override."""
        print(f"[ARMSTRONG] Override accepted: "
              f"Level {msg.override_level} -> {msg.selected_action}")
        return {
            "action":         msg.selected_action,
            "override_level": msg.override_level,
            "rationale":      msg.rationale,
            "timed_out":      False
        }

    def get_override_stats(self) -> dict:
        """Statistics on human override patterns."""
        if not self.override_history:
            return {"total": 0}

        total     = len(self.override_history)
        timeouts  = sum(1 for h in self.override_history if h["timed_out"])
        by_level  = {}
        for h in self.override_history:
            lvl = h["override_level"]
            by_level[lvl] = by_level.get(lvl, 0) + 1

        return {
            "total":          total,
            "timeouts":       timeouts,
            "timeout_rate":   timeouts / total,
            "by_level":       by_level,
            "override_rate":  (total - timeouts) / total
        }