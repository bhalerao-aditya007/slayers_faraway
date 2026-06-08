"""
Uncertainty-Aware Digital Twin.
Vectorized Monte-Carlo propagation using RK4.
"""

import numpy as np
from typing import Dict
from .physics import PhysicsSimulator


class DigitalTwin:
    def __init__(self, config, n_mc: int = 100):
        self.sim = PhysicsSimulator(config, n_mc=n_mc)
        self.n_mc = n_mc

    def propagate(self, initial_state: np.ndarray, action: str,
                  horizon: float, dt: float) -> Dict:
        """
        Propagate ensemble forward with fixed-step RK4.
        Returns collision probability, trajectories, and resource histories.
        """
        n_steps = int(horizon / dt)
        state = initial_state.copy()
        t = 0.0

        # History arrays
        traj = np.zeros((self.n_mc, n_steps, 3))
        resources = np.zeros((self.n_mc, n_steps, 5))  # soc, prop, o2, co2, h2o

        action_flags = {'action': action, 'vent_time': 0.0}

        for i in range(n_steps):
            # RK4 with quaternion re-normalization at each stage
            k1 = self.sim.derivatives(state, t, action_flags)

            s2 = state + 0.5 * dt * k1
            q = s2[:, self.sim.layout.idx_qw:self.sim.layout.idx_qw + 4]
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            s2[:, self.sim.layout.idx_qw:self.sim.layout.idx_qw + 4] = q
            k2 = self.sim.derivatives(s2, t + 0.5 * dt, action_flags)

            s3 = state + 0.5 * dt * k2
            q = s3[:, self.sim.layout.idx_qw:self.sim.layout.idx_qw + 4]
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            s3[:, self.sim.layout.idx_qw:self.sim.layout.idx_qw + 4] = q
            k3 = self.sim.derivatives(s3, t + 0.5 * dt, action_flags)

            s4 = state + dt * k3
            q = s4[:, self.sim.layout.idx_qw:self.sim.layout.idx_qw + 4]
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            s4[:, self.sim.layout.idx_qw:self.sim.layout.idx_qw + 4] = q
            k4 = self.sim.derivatives(s4, t + dt, action_flags)

            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Final normalize
            q = state[:, self.sim.layout.idx_qw:self.sim.layout.idx_qw + 4]
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            state[:, self.sim.layout.idx_qw:self.sim.layout.idx_qw + 4] = q

            if action == 'EMERGENCY_VENT':
                action_flags['vent_time'] += dt

            # Record
            traj[:, i, :] = state[:, self.sim.layout.idx_x:self.sim.layout.idx_x + 3]
            resources[:, i, 0] = state[:, self.sim.layout.idx_soc]
            resources[:, i, 1] = state[:, self.sim.layout.idx_m_prop]
            resources[:, i, 2] = state[:, self.sim.layout.idx_m_o2]
            resources[:, i, 3] = state[:, self.sim.layout.idx_m_co2]
            resources[:, i, 4] = state[:, self.sim.layout.idx_m_h2o]

            t += dt

        # Metrics
        min_dist = np.min(np.linalg.norm(traj, axis=2), axis=1)  # (n_mc,)
        collision = min_dist < self.sim.cfg.keepout_radius
        p_collision = float(np.mean(collision))

        final_soc = state[:, self.sim.layout.idx_soc]
        final_prop = state[:, self.sim.layout.idx_m_prop]

        return {
            'collision_probability': p_collision,
            'collision_probability_std': float(np.std(collision)),
            'min_distance_mean': float(np.mean(min_dist)),
            'min_distance_std': float(np.std(min_dist)),
            'final_soc_mean': float(np.mean(final_soc)),
            'final_soc_std': float(np.std(final_soc)),
            'final_prop_mean': float(np.mean(final_prop)),
            'final_prop_std': float(np.std(final_prop)),
            'trajectories': traj,
            'resources': resources,
            'dt': dt,
            'horizon': horizon
        }
