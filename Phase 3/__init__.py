"""
Phase 3: Digital Twin & Counterfactual Engine
Couples 6-DoF physics, habitat subsystems, and Monte-Carlo uncertainty propagation
for real-time spacecraft autonomy.
"""
from .physics import SpacecraftConfig, default_spacecraft_config, PhysicsSimulator
from .digital_twin import DigitalTwin
from .counterfactual import CounterfactualEngine
from .agent import ActionAgent
from .visualization import CounterfactualVisualizer

__all__ = [
    'SpacecraftConfig', 'default_spacecraft_config', 'PhysicsSimulator',
    'DigitalTwin', 'CounterfactualEngine', 'ActionAgent', 'CounterfactualVisualizer'
]
