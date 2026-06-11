<div align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzRmMGE3ZmEyZjFkOTdlOTVlNDhlZjBlYTVlMDBlZDMyYzMyZDhjNiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/3o7aD2saalEvpjtceQ/giphy.gif" width="100%" height="250" style="object-fit: cover; border-radius: 10px;">
  
  <br>
  
  <h1><a href="https://github.com/atharv1909/spacecraft_autonomy"><img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=700&size=35&duration=3000&pause=1000&color=00F0FF&center=true&vCenter=true&width=1000&height=80&lines=Problem+Statement:;Spacecraft+Autonomy+&+Pose+Estimation;Vision-Based+Rendezvous+Navigation" alt="Typing SVG" /></a></h1>
  
  <p><b>Next-Generation Mission Control Dashboard & Deep Learning Perception Engine</b></p>

  <div>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
    <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" />
    <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" />
    <img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" />
  </div>
  <br>
  <a href="https://spacecraft-autonomy-222404104450.us-central1.run.app">
    <img src="https://img.shields.io/badge/🚀_Live_Mission_Control_API-Online_Now-00F0FF?style=for-the-badge" alt="Live Demo" />
  </a>
</div>

<br>

## 🌌 Overview

When dealing with deep space autonomous docking, simply knowing the pose of a target satellite isn't enough—you need to know exactly how **trustworthy** that prediction is. 

This repository contains the complete stack for a **Spacecraft Autonomy Mission Control System**. It pairs a state-of-the-art `ResNet-50` deep learning perception engine with a strict, mathematically rigorous **Jensen Gain Uncertainty Monitor** utilizing Hopf Fibration grid anchors to evaluate network instability across rotational domains. 

In short: *If the neural network is guessing, the spacecraft refuses to dock.*

---

## ✨ Core Architecture

### 🧠 1. Perception Engine
- **Model:** 101MB `PoseNet_ResNet-50`
- **Activations:** `SiLU` mapping for smoother gradient flow across dark-space domain shifts.
- **Normalization:** Auto-detects real-world SunLAMP imagery vs. synthetic data to flawlessly normalize lighting across extreme space environments.

### 🛡️ 2. Uncertainty & Trust (Jensen Gain)
Standard pose estimation networks suffer from symmetry ambiguity. This system mitigates that by actively rotating the input along $N$ planar orientations, generating a normalized entropy curve. 
- **High Confidence (< 25°):** Stable internal feature maps. Prediction is rock solid.
- **Moderate Confidence (25° - 55°):** Acceptable prediction but handled with caution.
- **Untrustworthy (> 55°):** Network instability detected. Override required.

### 🛰️ 3. Mission Control Dashboard
A beautifully crafted React dashboard handling live telemetry, inference metrics, and protocol overrides.
- **Armstrong Protocol:** Allows mission control engineers to seamlessly override perception decisions dynamically if anomalies are detected in orbit.

---

## 🚀 Live Deployments

- **Frontend Application:** Deployed via Vercel for instant Edge distribution.
- **Backend Inference API:** Deployed as a scalable, 2GiB Dockerized container on **Google Cloud Run**.

---

## ⚙️ Installation & Usage

Because this repository contains a massive 101MB `.pt` Deep Learning model, **Git LFS** is required to clone it properly.

```bash
# 1. Install Git LFS
git lfs install

# 2. Clone the repository
git clone https://github.com/atharv1909/spacecraft_autonomy.git

# 3. Enter directory and setup environment
cd spacecraft_autonomy
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)

# 4. Install dependencies
pip install -r requirements_web.txt
```

### Running Locally

To spin up the Python perception backend server locally:
```bash
python interface/app.py
```

---

<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=600&size=20&duration=4000&pause=1000&color=FFFFFF&center=true&vCenter=true&width=800&height=40&lines=Ad+Astra+Per+Aspera;Failure+is+not+an+option." alt="Ad Astra" />
</div>
