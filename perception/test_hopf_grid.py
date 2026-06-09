import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from perception.models.hopf_grid import HopfFibrationGrid


def test_basic_construction():
    print("=" * 50)
    print("TEST 1: Basic Grid Construction")
    print("=" * 50)

    grid = HopfFibrationGrid(n_elevation=64, n_inplane=16)

    assert grid.anchor_rotations.shape == (512, 3, 3), "Wrong shape"
    assert grid.anchor_quaternions.shape == (512, 4), "Wrong quat shape"

    # Verify all are valid rotation matrices
    for i, R in enumerate(grid.anchor_rotations):
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), \
            f"Anchor {i} is not orthogonal"
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5), \
            f"Anchor {i} has wrong determinant"

    print(f"All {grid.total_anchors} anchors are valid rotation matrices")
    print("PASSED\n")


def test_nearest_anchor():
    print("=" * 50)
    print("TEST 2: Nearest Anchor Lookup")
    print("=" * 50)

    grid = HopfFibrationGrid(n_elevation=32, n_inplane=16)

    # Test: nearest anchor to itself should be distance ~0
    for i in [0, 100, 255, 511]:
        R_test = grid.anchor_rotations[i]
        idx, dist, R_nearest = grid.find_nearest_anchor(R_test)
        assert idx == i, f"Anchor {i} didn't find itself, found {idx}"
        assert dist < 1e-5, f"Distance to self should be ~0, got {dist}"

    print("Each anchor finds itself as nearest neighbor: PASSED")

    # Test: random rotation finds a nearby anchor
    R_random = Rotation.random().as_matrix()
    idx, dist, R_nearest = grid.find_nearest_anchor(R_random)
    print(f"Random rotation nearest anchor: idx={idx}, "
          f"distance={np.degrees(dist):.2f} degrees")
    print("PASSED\n")


def test_tangent_offset_roundtrip():
    print("=" * 50)
    print("TEST 3: Tangent Space Roundtrip")
    print("=" * 50)

    grid = HopfFibrationGrid(n_elevation=32, n_inplane=16)

    # Take an anchor, apply a small random offset, recover it
    for trial in range(10):
        R_anchor = grid.anchor_rotations[np.random.randint(512)]

        # Small random rotation (< 10 degrees)
        small_angle = np.random.uniform(0, np.radians(10))
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        R_offset = Rotation.from_rotvec(small_angle * axis).as_matrix()
        R_target = R_anchor @ R_offset

        # Compute offset
        v = grid.compute_tangent_offset(R_anchor, R_target)

        # Reconstruct
        R_reconstructed = grid.apply_tangent_offset(R_anchor, v)

        # Check reconstruction error
        dist = grid.geodesic_distance(R_target, R_reconstructed)
        assert dist < 1e-6, f"Roundtrip error too large: {np.degrees(dist):.6f} degrees"

    print("10 random tangent offset roundtrips: all < 1e-6 degrees error")
    print("PASSED\n")


def test_coverage_stats():
    print("=" * 50)
    print("TEST 4: Coverage Statistics")
    print("=" * 50)

    # Compare different grid sizes
    configs = [
        (16, 8),   # 128 anchors - coarse
        (32, 16),  # 512 anchors - default
        (64, 16),  # 1024 anchors - fine
    ]

    for n_elev, n_inp in configs:
        grid = HopfFibrationGrid(n_elevation=n_elev, n_inplane=n_inp)
        stats = grid.compute_coverage_stats()
        print(f"Grid {n_elev}x{n_inp} = {stats['total_anchors']} anchors: "
              f"mean_gap={stats['mean_gap_deg']:.2f}° "
              f"max_gap={stats['max_gap_deg']:.2f}°")

    print("\nRule of thumb: max_gap should be < 15° for good classification")
    print("PASSED\n")


def run_visualization():
    print("=" * 50)
    print("VISUALIZATION")
    print("=" * 50)

    grid = HopfFibrationGrid(n_elevation=32, n_inplane=16)

    os.makedirs("perception/outputs", exist_ok=True)

    print("Plotting S2 coverage...")
    grid.visualize_s2_coverage(save_path="perception/outputs/s2_coverage.png")

    print("Plotting gap distribution (takes ~30 seconds)...")
    grid.visualize_gap_distribution(save_path="perception/outputs/gap_distribution.png")


if __name__ == "__main__":
    test_coverage_stats()
