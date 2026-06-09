"""
Action Agent: Redis-integrated wrapper for Phase 3.
Subscribes to perception.out & cognition.out, publishes action.out.
"""

import json
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict
from .physics import SpacecraftConfig, PhysicsSimulator
from .counterfactual import CounterfactualEngine
from scipy.spatial.transform import Rotation as SciRotation

class ActionAgent:
    def __init__(self, config: SpacecraftConfig,
                 redis_host: str = 'localhost', redis_port: int = 6379,
                 standalone: bool = False, n_mc: int = 100):
        self.cfg = config
        self.standalone = standalone
        self.engine = CounterfactualEngine(config, n_mc=n_mc)
        self.latest_pose: Optional[Dict] = None
        self.latest_situation: Optional[Dict] = None

        if not standalone:
            try:
                import redis as redis_lib
                self.r = redis_lib.Redis(host=redis_host, port=redis_port,
                                         decode_responses=True)
                self.pubsub = self.r.pubsub()
                self.pubsub.subscribe('perception.out', 'cognition.out', 'human.in')
                print(f"[ActionAgent] Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                print(f"[ActionAgent] Redis connection failed ({e}), falling back to standalone.")
                self.standalone = True
                self.r = None
        else:
            self.r = None

    def process_one(self, message_json: str) -> Optional[Dict]:
        """Process a single message (for standalone or threaded loop)."""
        try:
            msg = json.loads(message_json)
        except json.JSONDecodeError:
            return None

        agent_id = msg.get('agent_id')
        mtype = msg.get('message_type')

        if agent_id == 'perception' and mtype == 'pose_estimate':
            self.latest_pose = msg['payload']
            return None  # Wait for situation or explicit trigger

        if agent_id == 'cognition' and mtype == 'situation_vector':
            self.latest_situation = msg['payload']
            return self._generate_and_publish()

        if agent_id == 'human' and mtype == 'override':
            # Armstrong Protocol: log and act (simplified)
            print(f"[ActionAgent] Human override received: {msg['payload']}")
            return None

        return None

    def _generate_and_publish(self) -> Optional[Dict]:
        if self.latest_pose is None:
            print("[ActionAgent] No pose available, skipping.")
            return None

        state0 = self._build_initial_state()
        situation = self.latest_situation or {}
        results = self.engine.evaluate_all_actions(state0, situation)
        payload = self._format_payload(results)

        if self.r:
            self.r.publish('action.out', json.dumps(payload))
            print(f"[ActionAgent] Published recommendation: {payload['payload']['recommended_action']}")

        return payload

    def _build_initial_state(self):
        pose = {
            'translation': self.latest_pose['pose']['t'],
            'quaternion': self._rotmat_to_quat(self.latest_pose['pose']['R']),
            'sigma_t': self.latest_pose['uncertainty'].get('sigma_t', 0.05),
            'hopf_grid': self.latest_pose['uncertainty'].get('hopf_grid')
        }
        habitat = {}
        if self.latest_situation and 'subsystem_states' in self.latest_situation:
            habitat = self.latest_situation['subsystem_states']
        return self.engine.twin.sim.initialize_state(pose, habitat)

    @staticmethod
    def _rotmat_to_quat(R_mat):
        R_arr = np.array(R_mat, dtype=float)
        q_scipy = SciRotation.from_matrix(R_arr).as_quat()  # [x,y,z,w]
        return np.roll(q_scipy, 1).tolist()  # [w,x,y,z]

    def _format_payload(self, results: list) -> Dict:
        actions_out = []
        for r in results:
            # Compute 95% CI approx as mean ± 1.96*std for collision
            p_mean = r['metrics']['tactical']['collision_probability']
            p_std = r['metrics']['tactical']['collision_probability_std']
            ci_low = max(0.0, p_mean - 1.96 * p_std)
            ci_high = min(1.0, p_mean + 1.96 * p_std)

            actions_out.append({
                'name': r['action'],
                'score': round(r['score'], 4),
                'metrics': {
                    'collision_probability': {
                        'mean': round(p_mean, 4),
                        'std': round(p_std, 4),
                        'ci_95': [round(ci_low, 4), round(ci_high, 4)]
                    },
                    'final_soc': {
                        'mean': round(r['metrics']['strategic']['final_soc_mean'], 4),
                        'std': round(r['metrics']['strategic']['final_soc_std'], 4)
                    }
                }
            })

        # Trajectory point-clouds for Phase 4 visualization
        viz_data = {}
        for r in results:
            viz_data[r['action']] = {
                'trajectory_mean': np.mean(r['metrics']['tactical']['trajectories'], axis=0).tolist(),
                'resource_mean': np.mean(r['metrics']['strategic']['resources'], axis=0).tolist()
            }

        return {
            'agent_id': 'action',
            'message_type': 'action_recommendation',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'payload': {
                'situation_id': (self.latest_situation or {}).get('situation_id', 'unknown'),
                'recommended_action': results[0]['action'],
                'all_actions': actions_out,
                'visualization_data': viz_data
            }
        }
