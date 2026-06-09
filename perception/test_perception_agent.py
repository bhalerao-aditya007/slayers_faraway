import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from perception.perception_agent import PerceptionAgent


def make_test_image(h=64, w=64):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_basic_output_schema():
    """
    Verify output has all required fields with correct types.
    This is the contract your teammates depend on.
    """
    print("=" * 55)
    print("TEST 1: Output Schema Validation")
    print("=" * 55)

    R_fixed = Rotation.random(random_state=42).as_matrix()
    t_fixed = np.array([2.5, -1.0, 8.0])

    def dummy_pose_fn(image):
        return R_fixed.copy(), t_fixed.copy()

    agent = PerceptionAgent(pose_fn=dummy_pose_fn, run_jensen_gain=False)
    image = make_test_image()
    output = agent.predict(image)

    # Check pose fields
    assert len(output.pose.R) == 3
    assert len(output.pose.R[0]) == 3
    assert len(output.pose.t) == 3
    assert len(output.pose.quaternion) == 4
    assert abs(sum(q**2 for q in output.pose.quaternion) - 1.0) < 1e-5, \
        "Quaternion not unit length"

    # Check uncertainty fields
    assert output.uncertainty.confidence_level in ("high", "moderate", "low")
    assert output.uncertainty.jensen_gain >= 0
    assert output.uncertainty.sigma_R_deg >= 0
    assert output.uncertainty.nearest_anchor_idx >= 0

    # Check metadata
    assert "timestamp" in output.metadata
    assert "processing_time_ms" in output.metadata

    print("All schema fields present and valid")
    print(f"Processing time: {output.metadata['processing_time_ms']} ms")
    print("PASSED\n")


def test_json_serialization():
    """
    Output must be fully JSON serializable for Redis pub/sub.
    """
    print("=" * 55)
    print("TEST 2: JSON Serialization (Redis compatibility)")
    print("=" * 55)

    import json

    R_fixed = Rotation.random(random_state=1).as_matrix()

    def dummy_fn(image):
        return R_fixed.copy(), np.array([1.0, 2.0, 3.0])

    agent = PerceptionAgent(pose_fn=dummy_fn, run_jensen_gain=False)
    output = agent.predict(make_test_image())

    json_str = output.to_json()
    parsed = json.loads(json_str)

    assert "pose" in parsed
    assert "uncertainty" in parsed
    assert "metadata" in parsed

    print("JSON output:")
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    print("\nJSON serialization PASSED\n")


def test_trustworthy_flag():
    """
    is_trustworthy must be False for low confidence.
    Orchestrator uses this for emergency fallback decisions.
    """
    print("=" * 55)
    print("TEST 3: Trustworthy Flag")
    print("=" * 55)

    R_a = Rotation.from_euler('y', 0, degrees=True).as_matrix()
    R_b = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    counter = [0]

    def confused_fn(image):
        counter[0] += 1
        R = R_a if counter[0] % 2 == 0 else R_b
        return R, np.zeros(3)

    agent = PerceptionAgent(pose_fn=confused_fn, run_jensen_gain=True)
    output = agent.predict(make_test_image())

    print(f"Jensen Gain: {output.uncertainty.jensen_gain:.2f}°")
    print(f"Confidence: {output.uncertainty.confidence_label}")
    print(f"Is trustworthy: {output.is_trustworthy}")
    assert not output.is_trustworthy, "Confused estimator should not be trustworthy"
    print("PASSED\n")


def test_full_pipeline_output():
    """
    Print a complete example output — what teammates actually receive.
    """
    print("=" * 55)
    print("TEST 4: Full Pipeline Output Example")
    print("=" * 55)

    R_true = Rotation.from_euler('xyz', [30, -20, 45], degrees=True).as_matrix()
    t_true = np.array([3.2, -0.8, 12.5])

    def realistic_fn(image):
        noise = Rotation.from_rotvec(np.random.randn(3) * np.radians(2)).as_matrix()
        return R_true @ noise, t_true + np.random.randn(3) * 0.1

    agent = PerceptionAgent(pose_fn=realistic_fn, run_jensen_gain=True)
    output = agent.predict(make_test_image())

    print("Complete PerceptionOutput (what Cognition agent receives):")
    print(output.to_json())
    print(f"\nis_trustworthy: {output.is_trustworthy}")
    print("PASSED\n")


if __name__ == "__main__":
    test_basic_output_schema()
    test_json_serialization()
    test_trustworthy_flag()
    test_full_pipeline_output()

    print("=" * 55)
    print("ALL PERCEPTION AGENT TESTS PASSED")
    print("Teammate interface is ready and validated")
    print("=" * 55)
