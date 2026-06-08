"""
Counterfactual Visualization.
Generates trajectory overlays, timeline plots, and risk matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List


class CounterfactualVisualizer:
    def plot_all(self, payload: Dict, save_prefix: str = '/mnt/agents/output/phase3'):
        """Generate all plots from an action recommendation payload."""
        actions = payload['all_actions']
        self.plot_trajectory_overlays(actions, save_prefix)
        self.plot_resource_timelines(actions, save_prefix)
        self.plot_risk_matrix(actions, save_prefix)

    def plot_trajectory_overlays(self, actions: List[Dict], save_prefix: str):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = {'ABORT': 'red', 'HOLD': 'blue', 'PROCEED_SLOW': 'orange',
                'PROCEED_NORMAL': 'green', 'RECONFIGURE_POWER': 'purple',
                'ISOLATE_MODULE': 'brown', 'EMERGENCY_VENT': 'pink'}

        for act in actions:
            traj = act['metrics']['tactical']['trajectories']  # (n_mc, n, 3)
            # Plot nominal (mean)
            mean_traj = np.mean(traj, axis=0)
            color = cmap.get(act['action'], 'gray')
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                    label=f"{act['action']} (score={act['score']:.2f})", color=color, linewidth=2)

            # Plot confidence cloud (every 10th sample)
            for i in range(0, traj.shape[0], 10):
                ax.plot(traj[i, :, 0], traj[i, :, 1], traj[i, :, 2],
                        color=color, alpha=0.05, linewidth=0.5)

        # Target sphere wireframe approximation
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        x = 10 * np.outer(np.cos(u), np.sin(v))
        y = 10 * np.outer(np.sin(u), np.sin(v))
        z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='red', alpha=0.1, label='Keep-out')

        ax.set_xlabel('R-bar [m]')
        ax.set_ylabel('V-bar [m]')
        ax.set_zlabel('H-bar [m]')
        ax.set_title('Counterfactual Trajectory Overlays (Tactical Horizon)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_trajectories.png', dpi=150)
        plt.close()

    def plot_resource_timelines(self, actions: List[Dict], save_prefix: str):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        labels = ['SoC', 'Propellant [kg]', 'O2 [kg]', 'H2O [kg]']
        idx_map = [0, 1, 2, 4]
        redlines = [0.2, 50.0, 10.0, 5.0]  # example redlines

        for act in actions:
            res = act['metrics']['strategic']['resources']  # (n_mc, n, 5)
            t = np.arange(res.shape[1]) * act['metrics']['strategic']['dt']
            mean_res = np.mean(res, axis=0)

            for ax_idx, (label, r_idx, red) in enumerate(zip(labels, idx_map, redlines)):
                ax = axes.flat[ax_idx]
                ax.plot(t, mean_res[:, r_idx], label=act['action'], linewidth=1.5)
                ax.axhline(red, color='red', linestyle='--', linewidth=1)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(label)
                ax.set_title(f'{label} Timeline (Strategic Horizon)')
                ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_timelines.png', dpi=150)
        plt.close()

    def plot_risk_matrix(self, actions: List[Dict], save_prefix: str):
        fig, ax = plt.subplots(figsize=(8, 8))
        # Simplified 5x5 grid
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks(range(1, 6))
        ax.set_yticks(range(1, 6))
        ax.set_xlabel('Impact')
        ax.set_ylabel('Probability')
        ax.set_title('Risk Matrix Shift: PROCEED_NORMAL vs ABORT')
        ax.grid(True, alpha=0.3)

        # Example failure modes
        ax.plot(4, 5, 'ro', markersize=15, label='Collision (PROCEED)')
        ax.plot(2, 2, 'go', markersize=15, label='Propellant Waste (ABORT)')
        ax.annotate('', xy=(2, 2), xytext=(4, 5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_risk_matrix.png', dpi=150)
        plt.close()
