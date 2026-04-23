<div align="center">

# Assistive Robot Teleoperation System
### Haptic Feedback · Shared Autonomy · Real-Time Computer Vision

**Refining Haptic Feedback and Cross-Platform Portability for Assistive Robot Teleoperation**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorRT](https://img.shields.io/badge/TensorRT-GPU-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Academic-orange)](LICENSE)
[![University](https://img.shields.io/badge/UTARI-UTA-003DA5)](https://www.uta.edu/research/institutes-and-centers/utari)

> **Developed at the University of Texas at Arlington Research Institute (UTARI)**  
> Department of Automation and Intelligent Systems · April 2026

</div>

---

## Project Overview
[▶️ Click to Watch Demo](https://drive.google.com/file/d/12WndrZ-ZZyDsB1BFT-VslWcZvzAbbmfS/view?usp=sharing)

This project presents an advanced **assistive robot teleoperation framework** designed to empower individuals who are **blind or visually impaired (BVI)** to safely and confidently control mobile robots. Traditional teleoperation systems create high cognitive load and anxiety for BVI users due to their reliance on visual feedback. This system solves that problem by fusing multi-modal sensory data into a rich, intuitive **tactile experience**.

The system makes two key contributions over prior art:

1. **Refined Haptic Feedback API** — Characterizes the PS5 DualSense controller's hardware and redesigns the feedback model to encode proximity, **direction**, and **urgency** as distinct tactile signals via separate vibration patterns.
2. **Cross-Platform Portability** — Adapts the existing ROS-based software from the KAIROS+ platform to the compact **ROSOrin robot**, refactoring all hardware-specific parameters into a flexible configuration layer deployable across different robotic platforms.

---

## Key Features

| Feature | Description |
|---|---|
| 🎮 **PS5 DualSense Integration** | Full joystick mapping with deadzone filtering, input smoothing, and deadman safety switch |
| 🛡️ **Shared Autonomy Control (SAC)** | Blends human joystick intent with LiDAR-driven Artificial Potential Field repulsion to prevent collisions in real time |
| 📳 **Sector-Based Haptic Feedback** | Directional, proximity-aware vibration patterns on the PS5 DualSense encode *where* and *how close* an obstacle is |
| 🧠 **Semantic Computer Vision** | Mask2Former deep learning model (TensorRT-optimized) classifies into Wall, Floor, Person, and Object in real time |
| 🔁 **Cross-Platform Architecture** | Fully Dockerized ROS 2 environment ensures reproducible deployment across machines and different robot hardware |
| 🤖 **Gazebo Simulation** | All modules validated in Gazebo simulation before deployment on physical hardware |
| 📡 **RViz2 Visualization** | Live marker overlays for obstacle clusters, repulsion vectors, human intent, and blended velocity commands |

---

## System Architecture

The system is a **bidirectional human–robot interaction framework** built on ROS 2 Humble. Three tightly integrated subsystems form the core pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                      OPERATOR (BVI User)                    │
│                  PS5 DualSense Controller                   │
└─────────────────┬───────────────────────┬───────────────────┘
                  │ Joystick Input        │ Haptic Output
                  ▼                       ▲
┌─────────────────────────────────────────────────────────────┐
│               SHARED AUTONOMY CONTROLLER (SAC)              │
│  LiDAR Scan → Frame Transform → Obstacle Detect → APF       │
│  Signal Blend (Human Intent + Repulsion) → Motor Output     │
└──────────┬────────────────────────────────────┬─────────────┘
           │                                    │
           ▼                                    ▼
┌──────────────────────┐          ┌─────────────────────────┐
│  COMPUTER VISION     │          │   HAPTIC FEEDBACK       │
│  Mask2Former + TRT   │          │   DualSense Rumble API  │
│  Wall/Floor/Person/  │          │   Zone × Sector Matrix  │
│  Object Segmentation │          │   Direction + Urgency   │
└──────────────────────┘          └─────────────────────────┘
           │                                    ▲
           └──────────────────┬─────────────────┘
                              ▼
                   ┌─────────────────┐
                   │  ROSOrin Robot  │
                   │  Mobile Base    │
                   └─────────────────┘
```

---

##  Module Deep Dives

<details>
<summary><b>🛡️ Shared Autonomy Controller (SAC)</b></summary>

The SAC is a **purely reactive** intermediary ROS 2 node that sits between the joystick driver and the robot's motion controller. It never plans paths — instead, every 10 Hz control cycle it:

1. Transforms LiDAR points from the sensor frame to the robot's `base_link` frame using a full 4×4 homogeneous transformation
2. Segments obstacle clusters from points within the `marker_threshold_range` (2.0 m)
3. Computes centroid and closest-point geometry for each cluster
4. Builds Region of Influence (ROI) boundaries expanded by ρ₀ = 0.75 m from each obstacle surface
5. Applies the **Artificial Potential Field (APF)** repulsion formula:

```
F = k_rep / ρ²  ×  (1/ρ − 1/ρ₀)   for ρ_cap ≤ ρ ≤ ρ₀
F = k_rep / ρ_cap²  ×  (1/ρ_cap − 1/ρ₀)   for ρ < ρ_cap  (capped)
F = 0   for ρ > ρ₀
```

6. Blends human joystick signal with the repulsion resultant: `v_final = (ref_signal + rep_resultant) / 2`
7. Gates all output through a **deadman switch** (buttons[10])

**Key Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `scan_threshold_range` | 5.0 m | Max range for obstacle consideration |
| `marker_threshold_range` | 2.0 m | Range to classify as obstacle |
| `rho_0 (ρ₀)` | 0.75 m | Region of Influence radius |
| `rho_cap` | 0.5 m | Singularity prevention cap |
| `k_rep` | 1.7 | APF repulsion scaling constant |
| `loop_frequency` | 10 Hz | Control cycle rate |

</details>

<details>
<summary><b>📳 Haptic Feedback System</b></summary>

The HapticController module encodes spatial obstacle information into **distinct tactile signals** on the PS5 DualSense's dual rumble motors, allowing BVI users to perceive direction and urgency without any visual feedback.

**Zone Definitions:**

| Zone | Distance | Behavior |
|---|---|---|
| ⚠️ Warning | 0.5 m – 0.25 m | Slow, rhythmic pulses |
| 🚨 Danger | < 0.25 m | Fast, urgent patterns |

**Sector-to-Motor Mapping:**

| Sector | Left Motor | Right Motor |
|---|---|---|
| Front | ✅ Active | ✅ Active (simultaneous) |
| Back | ✅ Active | ✅ Active (alternating) |
| Left | ✅ Active | ❌ Off |
| Right | ❌ Off | ✅ Active |

**Intensity Values (Warning / Danger):**

| Sector | Left Motor | Right Motor |
|---|---|---|
| Left | 25 / 100 | 0 / 0 |
| Right | 0 / 0 | 63 / 255 |
| Front / Back | 25 / 100 | 63 / 255 |

> Right motor values are intentionally higher to compensate for its smaller size requiring more drive to produce an equivalent physical sensation.

**Visual Indicator:** Controller LED switches from 🟢 Green (safe) → 🔴 Red (proximity breach).

</details>

<details>
<summary><b>🧠 Computer Vision Segmentation</b></summary>

The vision module uses **Mask2Former**, a state-of-the-art panoptic segmentation transformer, optimized with **TensorRT** for GPU execution. Every pixel in the camera feed is classified into one of four operational categories:

| Color | Category | ADE20K Sources |
|---|---|---|
| 🟡 Yellow | Wall | ID 0 |
| 🟢 Green | Floor | IDs 3, 28 (Floor, Rug) |
| 🔴 Red | Person | ID 12 |
| 🟣 Purple | Object | All other confident detections |

**Processing Pipeline:**

```
PIL Image (any size)
      ↓  Resize + Normalize + Reshape
Tensor (1, 3, 384, 384)
      ↓  TensorRT GPU Inference (async)
Raw Logits (class scores + mask predictions for 100 queries)
      ↓  Confidence filter (threshold 0.5) + Sigmoid + Pixel assignment
Label Map (96 × 96)
      ↓  Nearest-neighbor upscale to 512 × 512
Blended Overlay  +  Polygon Map
```

**Performance:** TensorRT delivers ~5–10× faster inference vs. native PyTorch, enabling near real-time video processing with pre-allocated GPU memory (zero per-frame allocation overhead).

</details>

<details>
<summary><b>🎮 PS5 DualSense Controller Node</b></summary>

| Function | Input | Notes |
|---|---|---|
| Safety Enable | R1 Trigger | Deadman switch — must be held |
| Mode Toggle | R1 + L1 | Mode 1: Differential, Mode 2: Holonomic |
| Linear Velocity | Left Stick Y | Max 0.3 m/s |
| Angular Velocity | Right Stick X | Max 0.8 rad/s |
| Lateral Velocity | Left Stick X | Mode 2 only |

Signal conditioning: ±0.25 deadzone filter + 30% linear interpolation smoothing to prevent velocity spikes.

</details>

---

## Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Robotics Middleware** | ROS 2 Humble |
| **Deep Learning** | Mask2Former (Panoptic Segmentation) |
| **GPU Inference** | NVIDIA TensorRT |
| **Computer Vision** | OpenCV, PIL |
| **Controller Interface** | pydualsense |
| **Sensor Fusion** | LiDAR (LaserScan) + RGB Camera |
| **Visualization** | RViz2 |
| **Simulation** | Gazebo |
| **Language** | Python 3.10 |

</div>

---

## Getting Started

### Prerequisites

```bash
# ROS 2 Humble
sudo apt install ros-humble-desktop

# Python dependencies
pip install pydualsense opencv-python Pillow numpy

# TensorRT (NVIDIA GPU required)
# Follow: https://developer.nvidia.com/tensorrt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/nguyenhuytrong/assistive-robot-teleoperation.git
cd assistive-robot-teleoperation

# Build the ROS 2 workspace
colcon build
source install/setup.bash

# Install libraries in requirement.txt
pip install -r requirement.txt
```

### Running the System

```bash
ros2 launch ps5 ps5.launch.oy
ros2 launch sac sac.launch.py
ros2 launch cv cv.launch.py
```

> **DualSense Setup:** Connect the controller via USB or Bluetooth. On first run, the lightbar should illuminate **green**. If it doesn't, check `/dev/hidraw` permissions:
> ```bash
> sudo chmod 666 /dev/hidraw*
> ```

---

##  Results

The system was evaluated on the **ROSOrin** mobile platform:

| Metric | Outcome |
|---|---|
| **Real-time performance** | ✅ Achieved — LiDAR + vision fusion at 10 Hz control cycle |
| **Obstacle avoidance** | ✅ SAC prevented collisions while preserving operator intent |
| **Haptic directional clarity** | ✅ Users correctly identified obstacle direction from tactile cues alone |
| **Vision segmentation** | ✅ Detected walls, floors, people, and objects in complex environments |
| **Cross-platform portability** | ✅ Successfully ported from KAIROS+ to ROSOrin with config-layer abstraction |

---



## License

This project was developed for academic research purposes at the University of Texas at Arlington.

