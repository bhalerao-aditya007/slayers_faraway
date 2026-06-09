import numpy as np
from scipy.spatial.transform import Rotation
from datetime import datetime, timezone
from typing import Callable, Optional
from dataclasses import dataclass, asdict
import json

from perception.models.hopf_grid import HopfFibrationGrid
from perception.models.jensen_gain import JensenGainMonitor


@dataclass
class PoseEstimate:
    """
    Rotation matrix (3x3) and translation vector (3,).
    Both in camera frame. Translation in meters.
    """
    R: list           # (3,3) as nested list for JSON serialization
    t: list           # (3,) as list
    quaternion: list  # (4,) [w, x, y, z]


@dataclass
class UncertaintyEstimate:
    """
    All uncertainty signals from the perception layer.
    This is what the Cognition agent uses to decide
    how much to trust the pose estimate.
    """
    jensen_gain: float        # degrees — main uncertainty signal
    confidence_level: str     # "high" / "moderate" / "low"
    confidence_label: str     # human readable string
    sigma_R_deg: float        # estimated rotation uncertainty (degrees)
    sigma_t_m: float          # estimated translation uncertainty (meters)
    nearest_anchor_idx: int   # which Hopf anchor was selected
    anchor_distance_deg: float  # how far true pose is from nearest anchor


@dataclass
class PerceptionOutput:
    """
    Complete output from the Perception Agent.
    This is the message posted to Redis channel: perception.out

    Teammates: consume this dict. Do not reach into
    internal model state directly.
    """
    pose: PoseEstimate
    uncertainty: UncertaintyEstimate
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @property
    def is_trustworthy(self) -> bool:
        """
        Quick boolean check for orchestrator:
        True if confidence is high or moderate.
        False if low — trigger conservative fallback.
        """
        return self.uncertainty.confidence_level in ("high", "moderate")

    @property
    def R_numpy(self) -> np.ndarray:
        """Get rotation matrix as numpy array."""
        return np.array(self.pose.R)

    @property
    def t_numpy(self) -> np.ndarray:
        """Get translation as numpy array."""
        return np.array(self.pose.t)


class PerceptionAgent:
    """
    Main perception agent class.

    In production: wraps the trained JEPA + pose head model.
    For testing/integration: accepts any callable as pose_fn.

    Usage (production):
        agent = PerceptionAgent(model_path="checkpoints/best.pt")
        output = agent.predict(image_array)

    Usage (testing without model):
        def dummy_fn(img): return np.eye(3), np.zeros(3)
        agent = PerceptionAgent(pose_fn=dummy_fn)
        output = agent.predict(image_array)
    """

    VERSION = "0.1.0"

    def __init__(self,
                 model_path: Optional[str] = None,
                 pose_fn: Optional[Callable] = None,
                 n_elevation: int = 64,
                 n_inplane: int = 16,
                 n_jensen_rotations: int = 16,
                 run_jensen_gain: bool = True):
        """
        Args:
            model_path: path to trained model checkpoint (.pt file)
                        When teammate trains the model, they provide this.
            pose_fn: callable (image -> R, t) for testing without model
                     Exactly one of model_path or pose_fn must be provided.
            n_elevation: Hopf grid elevation samples (default 64)
            n_inplane: Hopf grid in-plane rotations (default 16)
            n_jensen_rotations: how many rotations for Jensen Gain (default 16)
            run_jensen_gain: set False to skip uncertainty (faster, less safe)
        """
        if model_path is None and pose_fn is None:
            raise ValueError("Provide either model_path or pose_fn")

        self.run_jensen_gain = run_jensen_gain
        self._pose_fn = pose_fn

        # Build Hopf grid
        self.grid = HopfFibrationGrid(
            n_elevation=n_elevation,
            n_inplane=n_inplane
        )

        # Build Jensen Gain monitor
        self.jg_monitor = JensenGainMonitor(n_rotations=n_jensen_rotations)

        # Load model if path provided
        if model_path is not None:
            self._load_model(model_path)

        print(f"PerceptionAgent v{self.VERSION} ready")
        print(f"  Grid: {self.grid.total_anchors} anchors")
        print(f"  Jensen Gain: {'enabled' if run_jensen_gain else 'disabled'}")

    def _load_model(self, model_path: str):
        import torch
        self._model = torch.load(model_path, map_location='cpu')
        self._model.eval()
        self._pose_fn = self._model_inference
    def _model_inference(self, image: np.ndarray):
        import torch
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            R, t = self._model(tensor)
        return R.squeeze().numpy(), t.squeeze().numpy()
    def _pose_fn_wrapper(self, image: np.ndarray):
        """
        Calls either the loaded model or the provided test function.
        Always returns (R: ndarray (3,3), t: ndarray (3,))
        """
        if self._pose_fn is not None:
            result = self._pose_fn(image)
            if isinstance(result, tuple):
                R, t = result
            else:
                # Some test fns return just R
                R = result
                t = np.zeros(3)
            return np.array(R), np.array(t)
        else:
            raise RuntimeError("No pose function or model loaded")

    def _R_to_quaternion(self, R: np.ndarray) -> list:
        """Convert rotation matrix to [w, x, y, z] quaternion."""
        rot = Rotation.from_matrix(R)
        q = rot.as_quat()  # scipy: [x, y, z, w]
        return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    def _estimate_sigma_R(self, jensen_gain: float) -> float:
        """
        Convert Jensen Gain to rotation uncertainty estimate (degrees).
        Linear approximation: sigma_R ~ 0.6 * jensen_gain
        (empirical factor, to be calibrated on real data)
        """
        return 0.6 * jensen_gain

    def _estimate_sigma_t(self, jensen_gain: float,
                          t_magnitude: float) -> float:
        """
        Estimate translation uncertainty.
        Heuristic: scales with distance and Jensen Gain.
        Higher uncertainty in rotation -> higher uncertainty in translation.
        """
        return 0.05 * t_magnitude * (1 + jensen_gain / 10.0)

    def predict(self, image: np.ndarray) -> PerceptionOutput:
        """
        Main inference method. Call this from the Orchestrator.

        Args:
            image: numpy array (H, W) or (H, W, C), uint8 or float32

        Returns:
            PerceptionOutput — fully typed, JSON-serializable
        """
        t_start = datetime.now(timezone.utc)

        # Normalize image to float32 if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Get base pose estimate
        R, t = self._pose_fn_wrapper(image)

        # Find nearest Hopf anchor
        anchor_idx, anchor_dist, R_anchor = self.grid.find_nearest_anchor(R)

        # Run Jensen Gain uncertainty quantification
        if self.run_jensen_gain:
            def _pose_only(img):
                R_pred, _ = self._pose_fn_wrapper(img)
                return R_pred

            jg_result = self.jg_monitor.compute(
                pose_fn=_pose_only,
                image=image,
                compensate_inplane=True
            )
            jensen_gain = jg_result["jensen_gain"]
            confidence_level = jg_result["confidence_level"]
            confidence_label = jg_result["confidence_label"]
        else:
            jensen_gain = 0.0
            confidence_level = "high"
            confidence_label = "HIGH CONFIDENCE (Jensen Gain skipped)"

        t_end = datetime.now(timezone.utc)
        processing_ms = (t_end - t_start).total_seconds() * 1000

        t_magnitude = float(np.linalg.norm(t))

        output = PerceptionOutput(
            pose=PoseEstimate(
                R=R.tolist(),
                t=t.tolist(),
                quaternion=self._R_to_quaternion(R)
            ),
            uncertainty=UncertaintyEstimate(
                jensen_gain=float(jensen_gain),
                confidence_level=confidence_level,
                confidence_label=confidence_label,
                sigma_R_deg=self._estimate_sigma_R(jensen_gain),
                sigma_t_m=self._estimate_sigma_t(jensen_gain, t_magnitude),
                nearest_anchor_idx=int(anchor_idx),
                anchor_distance_deg=float(np.degrees(anchor_dist))
            ),
            metadata={
                "timestamp": t_start.isoformat(),
                "model_version": self.VERSION,
                "processing_time_ms": round(processing_ms, 2),
                "image_shape": list(image.shape),
                "grid_anchors": self.grid.total_anchors,
                "jensen_gain_enabled": self.run_jensen_gain
            }
        )

        return output
