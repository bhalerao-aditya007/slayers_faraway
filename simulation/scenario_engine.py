"""
Phase 6 — Simulation Engine
Runs scripted scenarios to test the full agentic pipeline
without real hardware or trained models.
"""

import time
import json
import threading
import redis
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from enum import Enum


class EventType(str, Enum):
    ANOMALY             = "anomaly"
    PERCEPTION_CHALLENGE = "perception_challenge"
    POWER_FAILURE       = "power_failure"
    COMMUNICATION_LOSS  = "communication_loss"
    SCENARIO_END        = "scenario_end"


class SubsystemStatus(str, Enum):
    NOMINAL   = "nominal"
    DEGRADED  = "degraded"
    CRITICAL  = "critical"
    FAILED    = "failed"


@dataclass
class ScenarioEvent:
    """A scripted event that fires at a specific time."""
    time_s:      float
    event_type:  str
    subsystem:   str = "none"
    severity:    str = SubsystemStatus.NOMINAL
    description: str = ""
    effect:      str = ""


@dataclass
class HabitatState:
    """Current state of all habitat subsystems."""
    # Life support
    o2_level:     float = 95.0   # percent
    co2_level:    float = 0.5    # percent
    pressure:     float = 101.3  # kPa
    temperature:  float = 22.0   # celsius
    humidity:     float = 45.0   # percent

    # Power
    battery_level:    float = 87.0  # percent
    solar_generation: float = 100.0 # percent
    power_load:       float = 65.0  # percent

    # Thermal
    radiator_efficiency: float = 100.0  # percent
    coolant_flow:        float = 100.0  # percent
    heat_load:           float = 40.0   # percent

    # Status flags
    life_support_status: str = SubsystemStatus.NOMINAL
    power_status:        str = SubsystemStatus.NOMINAL
    thermal_status:      str = SubsystemStatus.NOMINAL

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpacecraftState:
    """Current state of spacecraft pose and dynamics."""
    # Position relative to target (meters)
    x: float = 10.0
    y: float = 0.0
    z: float = 0.0

    # Velocity (m/s)
    vx: float = -0.1
    vy: float = 0.0
    vz: float = 0.0

    # Attitude (euler angles degrees, simplified)
    roll:  float = 0.0
    pitch: float = 0.0
    yaw:   float = 0.0

    # Lighting condition affects Jensen Gain
    lighting_condition: str = "nominal"  # nominal / glare / shadow
    jensen_gain_base:   float = 1.5      # base uncertainty

    def distance(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scenario:
    """Complete scenario definition."""
    name:         str
    description:  str
    duration_s:   float
    events:       List[ScenarioEvent] = field(default_factory=list)
    spacecraft:   SpacecraftState = field(default_factory=SpacecraftState)
    habitat:      HabitatState = field(default_factory=HabitatState)


class ScenarioEngine:
    """
    Runs scripted scenarios and publishes simulated
    agent messages to Redis channels.

    Acts as a fake perception + cognition + action agent
    so the Orchestrator can be tested end-to-end.
    """

    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 dt: float = 1.0):
        """
        Args:
            dt: simulation timestep in seconds
        """
        self.redis  = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.dt     = dt
        self._running = False

    def run_scenario(self, scenario: Scenario, speed: float = 1.0):
        """
        Run a scenario in real time (or faster with speed > 1).

        Args:
            scenario: the scenario to run
            speed:    1.0 = real time, 2.0 = 2x faster
        """
        print(f"\n{'='*55}")
        print(f"SCENARIO: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Duration: {scenario.duration_s}s")
        print(f"Events: {len(scenario.events)}")
        print(f"{'='*55}\n")

        self._running = True
        elapsed = 0.0
        sc = scenario.spacecraft
        hab = scenario.habitat
        fired_events = set()

        while self._running and elapsed <= scenario.duration_s:
            # Fire scripted events
            for i, event in enumerate(scenario.events):
                if i not in fired_events and elapsed >= event.time_s:
                    self._fire_event(event, sc, hab)
                    fired_events.add(i)

            # Update physics (simplified Hill's equations)
            self._update_spacecraft(sc)

            # Publish simulated perception message
            self._publish_perception(sc, elapsed)

            # Publish simulated cognition message
            self._publish_cognition(hab, elapsed)

            # Publish simulated action message
            self._publish_action(sc, hab, elapsed)

            # Print status every 10 seconds
            if int(elapsed) % 10 == 0:
                self._print_status(elapsed, sc, hab)

            elapsed += self.dt
            time.sleep(self.dt / speed)

        self._running = False
        print(f"\nScenario '{scenario.name}' complete")

    def stop(self):
        self._running = False

    def _fire_event(self, event: ScenarioEvent,
                    sc: SpacecraftState,
                    hab: HabitatState):
        """Apply event effects to simulation state."""
        print(f"\n[EVENT T+{event.time_s}s] {event.event_type.upper()}: "
              f"{event.description}")

        if event.event_type == EventType.ANOMALY:
            if event.subsystem == "thermal":
                hab.radiator_efficiency = 30.0
                hab.heat_load = 95.0
                hab.thermal_status = event.severity
                print(f"  -> Thermal: radiator efficiency dropped to 30%")

            elif event.subsystem == "power":
                hab.battery_level = max(20.0, hab.battery_level - 40.0)
                hab.solar_generation = 50.0
                hab.power_status = event.severity
                print(f"  -> Power: battery at {hab.battery_level}%")

            elif event.subsystem == "life_support":
                hab.o2_level = max(60.0, hab.o2_level - 25.0)
                hab.life_support_status = event.severity
                print(f"  -> Life support: O2 at {hab.o2_level}%")

        elif event.event_type == EventType.PERCEPTION_CHALLENGE:
            if event.effect == "jensen_gain_spike":
                sc.lighting_condition = "glare"
                sc.jensen_gain_base = 22.0
                print(f"  -> Perception: Jensen Gain spiked to {sc.jensen_gain_base}°")

        elif event.event_type == EventType.POWER_FAILURE:
            hab.battery_level = 15.0
            hab.solar_generation = 0.0
            hab.power_status = SubsystemStatus.CRITICAL
            print(f"  -> CRITICAL: Total power failure")

    def _update_spacecraft(self, sc: SpacecraftState):
        """
        Simplified orbital mechanics.
        Hill's equations for proximity operations.
        Just linear approach for demo.
        """
        sc.x += sc.vx * self.dt
        sc.y += sc.vy * self.dt
        sc.z += sc.vz * self.dt

        # Slowly recover lighting after glare
        if sc.lighting_condition == "glare":
            sc.jensen_gain_base = max(1.5, sc.jensen_gain_base - 0.5)
            if sc.jensen_gain_base <= 1.5:
                sc.lighting_condition = "nominal"

    def _publish_perception(self, sc: SpacecraftState, elapsed: float):
        """Publish simulated perception message."""
        import math
        import numpy as np

        # Jensen gain spikes based on lighting + distance
        dist = sc.distance()
        jg = sc.jensen_gain_base + (0.1 * max(0, dist - 5))
        jg = min(jg, 45.0)

        is_trustworthy = jg < 15.0
        confidence = "high" if jg < 5 else "moderate" if jg < 15 else "low"

        # Simple rotation matrix (identity + small perturbation)
        R = [[1,0,0],[0,1,0],[0,0,1]]

        msg = {
            "agent_id": "perception",
            "message_type": "pose_estimate",
            "timestamp": time.time(),
            "message_id": str(time.time_ns()),
            "R": R,
            "t": [sc.x, sc.y, sc.z],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "jensen_gain": round(jg, 2),
            "confidence_level": confidence,
            "confidence_label": f"{confidence.upper()} CONFIDENCE",
            "sigma_R_deg": round(jg * 0.6, 2),
            "sigma_t_m": round(0.05 * dist, 2),
            "nearest_anchor_idx": 0,
            "anchor_distance_deg": round(jg * 0.4, 2),
            "is_trustworthy": is_trustworthy,
            "processing_time_ms": 33.0,
            "image_shape": [1080, 1920, 3]
        }
        self.redis.publish("perception.out", json.dumps(msg))

    def _publish_cognition(self, hab: HabitatState, elapsed: float):
        """Publish simulated cognition message."""
        anomaly = (
            hab.thermal_status != SubsystemStatus.NOMINAL or
            hab.power_status != SubsystemStatus.NOMINAL or
            hab.life_support_status != SubsystemStatus.NOMINAL
        )

        severity = "nominal"
        anomaly_type = "none"
        novelty = 0.0
        recommended = "hold_position"

        if hab.thermal_status == SubsystemStatus.CRITICAL:
            severity = "critical"
            anomaly_type = "thermal_failure"
            novelty = 0.3
            recommended = "reconfigure_power"
        elif hab.power_status == SubsystemStatus.CRITICAL:
            severity = "critical"
            anomaly_type = "power_loss"
            novelty = 0.8  # Novel — triggers escalation
            recommended = "isolate_module"
        elif hab.life_support_status != SubsystemStatus.NOMINAL:
            severity = "critical"
            anomaly_type = "life_support_degraded"
            novelty = 0.9
            recommended = "await_human"

        msg = {
            "agent_id": "cognition",
            "message_type": "situation_vector",
            "timestamp": time.time(),
            "message_id": str(time.time_ns()),
            "situation_id": f"sit_{int(elapsed)}",
            "anomaly_detected": anomaly,
            "anomaly_type": anomaly_type,
            "anomaly_severity": severity,
            "novelty_score": novelty,
            "similar_case_id": "case_2847" if not anomaly else "",
            "similar_case_outcome": "success" if not anomaly else "",
            "recommended_action": recommended,
            "action_confidence": 0.91 if not anomaly else 0.6,
            "explanation": (
                f"Nominal operations" if not anomaly
                else f"{anomaly_type} detected ({severity})"
            )
        }
        self.redis.publish("cognition.out", json.dumps(msg))

    def _publish_action(self, sc: SpacecraftState,
                        hab: HabitatState, elapsed: float):
        """Publish simulated action recommendation."""
        dist = sc.distance()
        collision_prob = max(0.0, min(1.0, 0.01 * (5 - dist))) if dist < 5 else 0.0

        if dist > 5:
            action = "proceed_slow"
            score = 0.85
        elif dist > 2:
            action = "hold_position"
            score = 0.75
        else:
            action = "abort"
            score = 0.95

        # Override if habitat critical
        if (hab.power_status == SubsystemStatus.CRITICAL or
                hab.life_support_status != SubsystemStatus.NOMINAL):
            action = "hold_position"
            score = 0.9

        msg = {
            "agent_id": "action",
            "message_type": "action_recommendation",
            "timestamp": time.time(),
            "message_id": str(time.time_ns()),
            "primary_action": action,
            "primary_score": score,
            "collision_prob": round(collision_prob, 3),
            "mission_success_prob": round(score, 2),
            "resource_cost": 0.2,
            "alternatives": [
                {"action": "hold_position", "score": 0.7},
                {"action": "abort", "score": 0.5}
            ],
            "simulation_horizon_s": 60,
            "mc_runs": 100,
            "explanation": f"Distance {dist:.1f}m, collision_prob={collision_prob:.3f}"
        }
        self.redis.publish("action.out", json.dumps(msg))

    def _print_status(self, elapsed: float,
                      sc: SpacecraftState,
                      hab: HabitatState):
        """Print current simulation status."""
        print(f"[T+{elapsed:.0f}s] "
              f"dist={sc.distance():.1f}m "
              f"JG={sc.jensen_gain_base:.1f}° "
              f"O2={hab.o2_level:.0f}% "
              f"PWR={hab.battery_level:.0f}% "
              f"thermal={hab.thermal_status} "
              f"lighting={sc.lighting_condition}")