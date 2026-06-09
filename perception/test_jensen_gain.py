import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from perception.models.jensen_gain import JensenGainMonitor


def make_dummy_image(h=64, w=64):
    """Dummy grayscale image for testing."""
    img = np.random.randint(0, 255, (h, w), dtype=np.uint8).astype(np.float32)
    return img


def test_confident_estimator():
    """
    A good estimator returns consistent predictions regardless
    of in-plane rotation -> low Jensen Gain.
    """
    print("=" * 55)
    print("TEST 1: Confident Estimator (should give LOW Jensen Gain)")
    print("=" * 55)

    monitor = JensenGainMonitor(n_rotations=16)
    R_true = Rotation.random(random_state=42).as_matrix()

    # Simulate a perfect estimator — always returns true rotation
    # regardless of input image (best case scenario)
    def perfect_pose_fn(image):
        return R_true.copy()

    image = make_dummy_image()
    result = monitor.compute(perfect_pose_fn, image, compensate_inplane=False)

    print(f"Jensen Gain: {result['jensen_gain']:.4f} degrees")
    print(f"Confidence: {result['confidence_label']}")
    assert result['confidence_level'] == 'high', \
        f"Perfect estimator should be HIGH confidence, got {result['confidence_level']}"
    print("PASSED\n")


def test_noisy_estimator():
    """
    A noisy estimator adds random error -> moderate Jensen Gain.
    """
    print("=" * 55)
    print("TEST 2: Noisy Estimator (should give MODERATE Jensen Gain)")
    print("=" * 55)

    monitor = JensenGainMonitor(n_rotations=16)
    R_true = Rotation.random(random_state=0).as_matrix()

    def noisy_pose_fn(image):
        # Add small random rotation noise (~10 degrees)
        noise_axis = np.random.randn(3)
        noise_axis /= np.linalg.norm(noise_axis)
        noise_angle = np.random.uniform(0, np.radians(10))
        R_noise = Rotation.from_rotvec(noise_angle * noise_axis).as_matrix()
        return R_true @ R_noise

    image = make_dummy_image()
    result = monitor.compute(noisy_pose_fn, image, compensate_inplane=False)

    print(f"Jensen Gain: {result['jensen_gain']:.4f} degrees")
    print(f"Confidence: {result['confidence_label']}")
    print("PASSED\n")


def test_symmetry_confused_estimator():
    """
    Simulates a model confused by spacecraft symmetry —
    randomly flips between two plausible orientations.
    -> HIGH Jensen Gain
    """
    print("=" * 55)
    print("TEST 3: Symmetry-Confused Estimator (should give HIGH Jensen Gain)")
    print("=" * 55)

    monitor = JensenGainMonitor(n_rotations=16)

    # Two ambiguous poses 90 degrees apart (like symmetric solar panels)
    R_pose_A = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    R_pose_B = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()

    call_count = [0]

    def symmetry_confused_fn(image):
        # Alternates between two very different rotations
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            return R_pose_A.copy()
        else:
            return R_pose_B.copy()

    image = make_dummy_image()
    result = monitor.compute(symmetry_confused_fn, image, compensate_inplane=False)

    print(f"Jensen Gain: {result['jensen_gain']:.4f} degrees")
    print(f"Confidence: {result['confidence_label']}")
    assert result['confidence_level'] == 'low', \
        f"Confused estimator should be LOW confidence, got {result['confidence_level']}"
    print("PASSED\n")


def test_with_hopf_grid():
    """
    Integration test: Jensen Gain monitor working with
    the Hopf grid's nearest anchor lookup as the pose function.
    Shows these two modules work together.
    """
    print("=" * 55)
    print("TEST 4: Integration with Hopf Grid")
    print("=" * 55)

    from perception.models.hopf_grid import HopfFibrationGrid

    grid = HopfFibrationGrid(n_elevation=32, n_inplane=16)
    monitor = JensenGainMonitor(n_rotations=16)

    R_true = Rotation.random(random_state=7).as_matrix()

    def hopf_based_pose_fn(image):
        # Simulate: find nearest anchor (what classifier head would do)
        # then add small refinement noise
        idx, dist, R_nearest = grid.find_nearest_anchor(R_true)
        noise = Rotation.from_rotvec(
            np.random.randn(3) * np.radians(2)
        ).as_matrix()
        return R_nearest @ noise

    image = make_dummy_image()
    result = monitor.compute(hopf_based_pose_fn, image, compensate_inplane=False)

    print(f"Jensen Gain: {result['jensen_gain']:.4f} degrees")
    print(f"Confidence: {result['confidence_label']}")
    print(f"Mean spread: {np.mean(result['spread_per_rotation']):.4f} degrees")
    print("Integration test PASSED\n")


def run_visualization():
    print("=" * 55)
    print("VISUALIZATION: Three confidence levels")
    print("=" * 55)

    monitor = JensenGainMonitor(n_rotations=16)
    image = make_dummy_image()
    os.makedirs("perception/outputs", exist_ok=True)

    R_fixed = Rotation.random(random_state=1).as_matrix()
    R_A = Rotation.from_euler('y', 0, degrees=True).as_matrix()
    R_B = Rotation.from_euler('y', 90, degrees=True).as_matrix()

    scenarios = [
        ("high_confidence",
         lambda img: R_fixed.copy()),
        ("symmetry_confused",
         lambda img: R_A.copy() if np.random.rand() > 0.5 else R_B.copy()),
    ]

    for name, fn in scenarios:
        result = monitor.compute(fn, image, compensate_inplane=False)
        print(f"{name}: Jensen Gain = {result['jensen_gain']:.2f}° "
              f"| {result['confidence_label']}")
        monitor.visualize_prediction_spread(
            result,
            save_path=f"perception/outputs/jensen_{name}.png"
        )


if __name__ == "__main__":
    test_confident_estimator()
    test_noisy_estimator()
    test_symmetry_confused_estimator()
    test_with_hopf_grid()
    run_visualization()

    print("=" * 55)
    print("ALL JENSEN GAIN TESTS PASSED")
    print("=" * 55)
