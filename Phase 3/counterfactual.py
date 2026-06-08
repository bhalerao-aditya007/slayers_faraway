"""
Counterfactual Scenario Generator.
Multi-horizon, multi-objective action scoring.
"""

import numpy as np
from typing import Dict, List
from .digital_twin import DigitalTwin


class CounterfactualEngine:
    ACTIONS = [
        'ABORT', 'HOLD', 'PROCEED_SLOW', 'PROCEED_NORMAL',
        'RECONFIGURE_POWER', 'ISOLATE_MODULE', 'EMERGENCY_VENT'
    ]

    HORIZONS = {
        'tactical': (60.0, 0.1),      # 1 min, 0.1 s
        'operational': (600.0, 1.0),  # 10 min, 1.0 s
        'strategic': (3600.0, 10.0)   # 1 hr, 10 s
    }

    def __init__(self, config, n_mc: int = 100):
        self.twin = DigitalTwin(config, n_mc=n_mc)
        self.cfg = config

    def evaluate_all_actions(self, initial_state: np.ndarray,
                             situation: Dict) -> List[Dict]:
        results = []
        for action in self.ACTIONS:
            metrics = {}
            for h_name, (T, dt) in self.HORIZONS.items():
                # Habitat actions only affect strategic/operational; skip tactical if irrelevant
                if action in ('RECONFIGURE_POWER', 'ISOLATE_MODULE', 'EMERGENCY_VENT') and h_name == 'tactical':
                    # Still run but with minimal horizon for consistency
                    T, dt = 10.0, 0.1
                m = self.twin.propagate(initial_state.copy(), action, T, dt)
                metrics[h_name] = m

            score = self._score_action(action, metrics, situation)
            results.append({
                'action': action,
                'score': score,
                'metrics': metrics
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def _score_action(self, action: str, metrics: Dict, situation: Dict) -> float:
        # Safety: use tactical collision probability
        p_col = metrics['tactical']['collision_probability']

        # Mission progress: operational final vs initial distance
        op = metrics['operational']
        traj = op['trajectories']  # (n_mc, n_steps, 3)
        initial_dist = np.mean(np.linalg.norm(traj[:, 0, :], axis=1))
        final_dist = np.mean(np.linalg.norm(traj[:, -1, :], axis=1))
        mission = np.clip((initial_dist - final_dist) / max(initial_dist, 1.0), 0.0, 1.0)

        # Resources: strategic remaining fractions
        st = metrics['strategic']
        soc_frac = st['final_soc_mean']
        prop_frac = st['final_prop_mean'] / self.cfg.initial_propellant
        resources = np.clip(soc_frac * prop_frac, 0.0, 1.0)

        # Structural stress proxy: max acceleration from thrusters
        # (simplified; real system would use FEM stress model)
        stress_ratio = 0.0
        safety = 1.0 - p_col - max(0.0, stress_ratio)
        safety = np.clip(safety, 0.0, 1.0)

        # Dynamic weight adaptation per architecture spec
        if p_col > 0.05:
            w_s, w_m, w_r = 0.9, 0.1, 0.0
        else:
            w_s, w_m, w_r = 0.4, 0.4, 0.2

        score = w_s * safety + w_m * mission + w_r * resources
        return float(score)
