import numpy as np
from scipy.spatial.transform import Rotation
from typing import Callable, Dict
import matplotlib
import matplotlib.pyplot as plt


class JensenGainMonitor:
    """
    Real-time uncertainty quantification for pose estimation.

    Core idea:
    A pose estimator f maps image -> rotation matrix.
    If f is highly sensitive to small symmetry-related image
    transformations (in-plane rotations), the predictions will
    be inconsistent = HIGH UNCERTAINTY.

    Jensen Gain measures this inconsistency:
        G_J = E[geodesic(f(T_k(x)), mean_pred)] over k transformations

    Low G_J  -> predictions are consistent across transformations -> confident
    High G_J -> predictions scatter wildly -> uncertain (symmetry ambiguity
                or out-of-distribution input)

    Thresholds (from outline):
        G_J < 0.5  : HIGH CONFIDENCE   (green)
        0.5 - 2.0  : MODERATE          (yellow)
        G_J >= 2.0 : LOW CONFIDENCE    (red)

    Note: thresholds are in DEGREES here for human interpretability.
    The outline values (0.5, 2.0) assume a normalized metric.
    We use degrees directly and set equivalent thresholds.
    """

    # Confidence thresholds in degrees
    HIGH_CONFIDENCE_THRESH = 5.0    # degrees
    MODERATE_THRESH = 15.0          # degrees

    CONFIDENCE_LEVELS = {
        "high": "HIGH CONFIDENCE",
        "moderate": "MODERATE",
        "low": "LOW CONFIDENCE / SYMMETRY AMBIGUITY"
    }

    def __init__(self, n_rotations: int = 16):
        """
        Args:
            n_rotations: number of in-plane rotations to sample
                         More = more accurate estimate, more compute
                         16 is the value from the outline
        """
        self.n_rotations = n_rotations
        self.angles_deg = np.linspace(0, 360, n_rotations, endpoint=False)
        self.angles_rad = np.radians(self.angles_deg)

    def _rotate_image_inplane(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Apply in-plane (roll) rotation to image.
        This simulates spacecraft rolling around camera optical axis —
        a transformation the pose estimator should handle consistently
        if it truly understands geometry.

        For numpy arrays we use scipy rotation on the image plane.
        In real code this will use cv2.warpAffine.

        Args:
            image: (H, W) or (H, W, C) numpy array
            angle_deg: rotation angle in degrees
        Returns:
            rotated image same shape
        """
        from scipy.ndimage import rotate as ndimage_rotate
        if image.ndim == 2:
            return ndimage_rotate(image, angle_deg, reshape=False, order=1)
        else:
            # Rotate spatial dims only, keep channels
            rotated = ndimage_rotate(image, angle_deg,
                                     axes=(0, 1), reshape=False, order=1)
            return rotated

    def _geodesic_mean(self, rotations: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """
        Compute the Frechet mean (geodesic mean) of a set of rotation matrices.

        Simple iterative algorithm:
        1. Start with first rotation as estimate
        2. Compute mean of Lie algebra offsets to all rotations
        3. Update estimate by applying mean offset
        4. Repeat until convergence

        Args:
            rotations: (N, 3, 3) array of rotation matrices
        Returns:
            R_mean: (3, 3) mean rotation matrix
        """
        R_mean = rotations[0].copy()

        for _ in range(max_iter):
            # Compute tangent vectors from current mean to each rotation
            tangents = []
            for R in rotations:
                R_rel = R_mean.T @ R
                rotvec = Rotation.from_matrix(R_rel).as_rotvec()
                tangents.append(rotvec)

            tangents = np.array(tangents)  # (N, 3)
            mean_tangent = tangents.mean(axis=0)

            # Check convergence
            if np.linalg.norm(mean_tangent) < 1e-8:
                break

            # Update mean
            R_update = Rotation.from_rotvec(mean_tangent).as_matrix()
            R_mean = R_mean @ R_update

        return R_mean

    def _geodesic_distance_deg(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Geodesic distance between two rotations in degrees."""
        R_rel = R1.T @ R2
        trace_val = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
        return np.degrees(np.arccos(trace_val))

    def compute(self,
                pose_fn: Callable,
                image: np.ndarray,
                compensate_inplane: bool = True) -> Dict:
        """
        Compute Jensen Gain for a single image.

        Args:
            pose_fn: callable that takes image array -> (3,3) rotation matrix
                     This is your perception model's inference function.
                     For testing, pass a lambda or dummy function.
            image: numpy array (H, W) or (H, W, C)
            compensate_inplane: if True, undo the applied rotation before
                                 computing spread (more accurate measure
                                 of estimator consistency)
        Returns:
            dict with:
                jensen_gain: float (degrees) — main uncertainty signal
                confidence_level: str — "high" / "moderate" / "low"
                confidence_label: str — human readable
                predictions: list of (3,3) rotation matrices
                mean_rotation: (3,3) geodesic mean of predictions
                spread_per_rotation: list of floats (degrees) per variant
        """
        predictions = []
        compensation_rotations = []

        for angle_deg in self.angles_deg:
            # Apply in-plane rotation to image
            img_rotated = self._rotate_image_inplane(image, angle_deg)

            # Get pose prediction
            R_pred = pose_fn(img_rotated)

            if compensate_inplane:
                # The image was rotated by angle_deg in-plane.
                # A perfect estimator would return R_pred = R_true @ R_inplane(angle)
                # We undo the in-plane component to compare in common frame.
                angle_rad = np.radians(angle_deg)
                c, s = np.cos(angle_rad), np.sin(angle_rad)
                R_inplane = np.array([
                    [c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]
                ])
                R_compensated = R_pred @ R_inplane.T
                predictions.append(R_compensated)
            else:
                predictions.append(R_pred)

            compensation_rotations.append(angle_deg)

        predictions = np.array(predictions)  # (N, 3, 3)

        # Compute geodesic mean of all predictions
        R_mean = self._geodesic_mean(predictions)

        # Jensen Gain = mean geodesic distance of each prediction from mean
        spreads = []
        for R_pred in predictions:
            dist = self._geodesic_distance_deg(R_pred, R_mean)
            spreads.append(dist)

        jensen_gain = np.mean(spreads)

        # Classify confidence
        if jensen_gain < self.HIGH_CONFIDENCE_THRESH:
            confidence_level = "high"
        elif jensen_gain < self.MODERATE_THRESH:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        return {
            "jensen_gain": jensen_gain,
            "confidence_level": confidence_level,
            "confidence_label": self.CONFIDENCE_LEVELS[confidence_level],
            "predictions": predictions,
            "mean_rotation": R_mean,
            "spread_per_rotation": spreads,
            "angles_tested_deg": self.angles_deg.tolist()
        }

    def visualize_prediction_spread(self,
                                    result: Dict,
                                    true_rotation: np.ndarray = None,
                                    save_path: str = None):
        """
        Visualize how spread out the predictions are across
        in-plane rotation variants.

        A tight cluster = low Jensen Gain = high confidence.
        A scattered spread = high Jensen Gain = symmetry ambiguity.
        """
        spreads = result["spread_per_rotation"]
        angles = result["angles_tested_deg"]
        jg = result["jensen_gain"]
        level = result["confidence_level"]

        color_map = {"high": "green", "moderate": "orange", "low": "red"}
        bar_color = color_map[level]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: spread per rotation angle
        ax1.bar(angles, spreads, width=360 / len(angles) * 0.8,
                color=bar_color, alpha=0.8, edgecolor='white')
        ax1.axhline(jg, color='black', linestyle='--',
                    label=f'Jensen Gain: {jg:.2f}°')
        ax1.axhline(self.HIGH_CONFIDENCE_THRESH, color='green',
                    linestyle=':', alpha=0.7, label=f'High conf threshold: {self.HIGH_CONFIDENCE_THRESH}°')
        ax1.axhline(self.MODERATE_THRESH, color='orange',
                    linestyle=':', alpha=0.7, label=f'Moderate threshold: {self.MODERATE_THRESH}°')
        ax1.set_xlabel('In-plane Rotation Applied (degrees)')
        ax1.set_ylabel('Geodesic Distance from Mean (degrees)')
        ax1.set_title(f'Prediction Spread per Rotation Variant\n'
                      f'Status: {result["confidence_label"]}')
        ax1.legend(fontsize=8)

        # Plot 2: polar plot of spread
        angles_rad = np.radians(angles + [angles[0]])
        spreads_polar = spreads + [spreads[0]]

        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles_rad, spreads_polar, color=bar_color, linewidth=2)
        ax2.fill(angles_rad, spreads_polar, color=bar_color, alpha=0.3)
        ax2.set_title(f'Polar View of Prediction Spread\nJensen Gain: {jg:.2f}°',
                      pad=20)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
        plt.close()
