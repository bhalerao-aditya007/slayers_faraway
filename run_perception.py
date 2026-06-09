#!/usr/bin/env python3
"""
Perception Agent runner — reads camera/images, publishes to Redis.
Replace the image source with your actual camera feed.
"""
import time
import json
import numpy as np
import redis
from perception.perception_agent import PerceptionAgent

# --- CONFIG ---
MODEL_PATH = "/path/to/your_model.pt"   # <-- YOUR MODEL HERE
REDIS_HOST = "localhost"
REDIS_PORT = 6379
USE_DUMMY  = False  # Set True to test WITHOUT your model

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

if USE_DUMMY:
    from scipy.spatial.transform import Rotation
    def dummy_fn(img):
        return Rotation.random().as_matrix(), np.array([5.0, 0.0, 0.0])
    agent = PerceptionAgent(pose_fn=dummy_fn, run_jensen_gain=True)
else:
    agent = PerceptionAgent(model_path=MODEL_PATH, run_jensen_gain=True)

print("Perception agent running. Ctrl+C to stop.")

while True:
    # --- REPLACE THIS with your actual image source ---
    # From camera:     image = your_camera.read()
    # From file:       image = cv2.imread("frame.jpg")
    # For testing:
    image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    # --------------------------------------------------

    output = agent.predict(image)
    
    # Build the Redis message matching PoseEstimateMessage schema
    msg = {
        "agent_id": "perception",
        "message_type": "pose_estimate",
        "timestamp": time.time(),
        "message_id": str(time.time_ns()),
        "R": output.pose.R,
        "t": output.pose.t,
        "quaternion": output.pose.quaternion,
        "jensen_gain": output.uncertainty.jensen_gain,
        "confidence_level": output.uncertainty.confidence_level,
        "confidence_label": output.uncertainty.confidence_label,
        "sigma_R_deg": output.uncertainty.sigma_R_deg,
        "sigma_t_m": output.uncertainty.sigma_t_m,
        "nearest_anchor_idx": output.uncertainty.nearest_anchor_idx,
        "anchor_distance_deg": output.uncertainty.anchor_distance_deg,
        "is_trustworthy": output.is_trustworthy,
        "processing_time_ms": output.metadata["processing_time_ms"],
        "image_shape": output.metadata["image_shape"]
    }
    
    r.publish("perception.out", json.dumps(msg))
    print(f"Published: JG={output.uncertainty.jensen_gain:.2f}° "
          f"trust={output.is_trustworthy} "
          f"conf={output.uncertainty.confidence_level}")
    
    time.sleep(0.5)  # 2 Hz, adjust as needed
