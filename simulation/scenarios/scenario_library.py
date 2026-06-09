"""
Pre-built scenario library.
These are the test scenarios from the outline Phase 6.4
"""

from simulation.scenario_engine import (
    Scenario, ScenarioEvent, SpacecraftState,
    HabitatState, EventType, SubsystemStatus
)


def nominal_docking() -> Scenario:
    """
    Routine docking — no failures, clean approach.
    Expected: Orchestrator maintains PROCEED_SLOW throughout.
    Jensen Gain stays low. No escalation.
    """
    return Scenario(
        name="Nominal Docking",
        description="Clean approach and docking, no anomalies",
        duration_s=120,
        spacecraft=SpacecraftState(
            x=15.0, vx=-0.1,
            jensen_gain_base=1.5
        ),
        habitat=HabitatState(),
        events=[]
    )


def thermal_anomaly() -> Scenario:
    """
    Single failure — thermal system degrades mid-approach.
    Expected: Orchestrator switches to RECONFIGURE_POWER,
    escalates to human.
    """
    return Scenario(
        name="Thermal Anomaly",
        description="Radiator coolant leak detected during approach",
        duration_s=180,
        spacecraft=SpacecraftState(x=12.0, vx=-0.1),
        habitat=HabitatState(),
        events=[
            ScenarioEvent(
                time_s=60,
                event_type=EventType.ANOMALY,
                subsystem="thermal",
                severity=SubsystemStatus.CRITICAL,
                description="Radiator 2 coolant leak detected"
            )
        ]
    )


def perception_challenge() -> Scenario:
    """
    Sun enters camera FOV — Jensen Gain spikes.
    Expected: Orchestrator forces HOLD, escalates.
    Jensen Gain monitor triggers uncertainty alert.
    """
    return Scenario(
        name="Perception Challenge",
        description="Sun glare causes pose uncertainty spike",
        duration_s=120,
        spacecraft=SpacecraftState(x=8.0, vx=-0.05),
        habitat=HabitatState(),
        events=[
            ScenarioEvent(
                time_s=30,
                event_type=EventType.PERCEPTION_CHALLENGE,
                description="Sun enters camera FOV",
                effect="jensen_gain_spike"
            )
        ]
    )


def perfect_storm() -> Scenario:
    """
    Compound failure — thermal + power + perception challenge.
    The most demanding scenario. Tests system under
    simultaneous multi-domain stress.
    Expected: Conservative HOLD, multiple escalations,
    high novelty score from cognition agent.
    """
    return Scenario(
        name="Perfect Storm",
        description="Thermal failure + power loss + pose uncertainty simultaneously",
        duration_s=300,
        spacecraft=SpacecraftState(x=10.0, vx=-0.08),
        habitat=HabitatState(),
        events=[
            ScenarioEvent(
                time_s=60,
                event_type=EventType.ANOMALY,
                subsystem="thermal",
                severity=SubsystemStatus.CRITICAL,
                description="Radiator failure"
            ),
            ScenarioEvent(
                time_s=90,
                event_type=EventType.PERCEPTION_CHALLENGE,
                description="Sun glare + thermal shimmer",
                effect="jensen_gain_spike"
            ),
            ScenarioEvent(
                time_s=120,
                event_type=EventType.POWER_FAILURE,
                subsystem="power",
                severity=SubsystemStatus.CRITICAL,
                description="Solar array partial failure"
            )
        ]
    )