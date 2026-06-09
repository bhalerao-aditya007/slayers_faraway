#!/usr/bin/env python3
"""
Standalone demonstration of Phase 3.
Simulates Perception + Cognition messages, runs counterfactuals, and generates plots.
"""

import json
import numpy as np
from phase3 import default_spacecraft_config, ActionAgent, CounterfactualVisualizer


def main():
    print("=== Phase 3 Digital Twin & Counterfactual Engine Demo ===\n")

    # 1. Build spacecraft configuration
    cfg = default_spacecraft_config()

    # 2. Instantiate agent in standalone mode (no Redis required)
    agent = ActionAgent(cfg, standalone=True, n_mc=50)

    # 3. Simulate a Perception message (pose estimate with Hopf uncertainty)
    #    Spacecraft is 50 m down R-bar, slight V-bar offset, uncertain attitude
    pose_msg = {
        'agent_id': 'perception',
        'message_type': 'pose_estimate',
        'timestamp': '2026-06-08T17:55:00Z',
        'payload': {
            'pose': {
                'R': np.eye(3).tolist(),
                't': [50.0, 5.0, 0.0]
            },
            'uncertainty': {
                'sigma_R': 0.12,
                'sigma_t': 0.05,
                'jensen_gain': 2.8,
                'hopf_grid': {
                    'anchors': [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.7071, 0.7071, 0.0, 0.0],
                        [0.7071, 0.0, 0.7071, 0.0]
                    ],
                    'probabilities': [0.6, 0.25, 0.15],
                    'offsets': [
                        [0.01, 0.0, 0.0],
                        [0.0, 0.02, 0.0],
                        [0.0, 0.0, 0.01]
                    ]
                }
            },
            'confidence': 'low'
        }
    }

    # 4. Simulate a Cognition message (thermal anomaly during approach)
    situation_msg = {
        'agent_id': 'cognition',
        'message_type': 'situation_vector',
        'timestamp': '2026-06-08T17:55:01Z',
        'payload': {
            'situation_id': 'sit-001',
            'anomaly': {
                'type': 'thermal_failure',
                'severity': 'critical',
                'subsystem': 'thermal'
            },
            'mission_phase': 'approach',
            'confidence': 'low',
            'subsystem_states': {
                'thermal': [310.0, 295.0, 270.0],  # Node temps [K]
                'soc': 0.75,
                'propellant': 480.0
            }
        }
    }

    # 5. Inject messages into agent
    print("Injecting simulated Perception message...")
    agent.process_one(json.dumps(pose_msg))

    print("Injecting simulated Cognition message...")
    result = agent.process_one(json.dumps(situation_msg))

    # 6. Display results
    if result:
        print("\n--- ACTION RECOMMENDATION ---")
        print(json.dumps(result['payload'], indent=2))

        # 7. Generate visualizations
        print("\nGenerating plots...")
        viz = CounterfactualVisualizer()
        viz.plot_all(result['payload'])
        print("Plots saved to /mnt/agents/output/phase3_*.png")
    else:
        print("No result generated.")


if __name__ == '__main__':
    main()
