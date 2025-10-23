# LeRobot Integration with Berkeley Humanoid Lite

Complete guide for integrating Hugging Face LeRobot framework with Berkeley Humanoid Lite for imitation learning-based robot control.

## Table of Contents

1. [Overview](#overview)
2. [What is LeRobot](#what-is-lerobot)
3. [Available Policy Models](#available-policy-models)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Understanding LeRobot Dataset Format](#understanding-lerobot-dataset-format)
7. [Phase 1: Camera Setup](#phase-1-camera-setup)
8. [Phase 2: Data Collection](#phase-2-data-collection)
9. [Phase 3: Training](#phase-3-training)
10. [Phase 4: Evaluation](#phase-4-evaluation)
11. [Phase 5: Deployment](#phase-5-deployment)
12. [Model Comparison Guide](#model-comparison-guide)
13. [Performance Optimization](#performance-optimization)
14. [Troubleshooting](#troubleshooting)
15. [LeRobot vs RL vs Groot N1](#lerobot-vs-rl-vs-groot-n1)

## Overview

### What This Integration Achieves

Transform Berkeley Humanoid Lite from RL-based policies to **imitation learning** policies that:

- Learn directly from human demonstrations (teleoperation)
- Require no reward engineering
- Generalize to similar tasks with few examples
- Use vision for perception and decision-making
- Support multiple policy architectures (ACT, Diffusion, etc.)
- Leverage Hugging Face ecosystem for sharing models/datasets

### Key Differences from Other Approaches

**vs Current RL (RSL-RL/PPO):**
- No reward function needed → Learn from demos
- Faster data collection → Hours vs days of simulation
- More intuitive → Show the robot what to do
- Vision-based → Can see and react to environment

**vs Groot N1:**
- Smaller models → Faster inference, less memory
- No language conditioning → Task-specific policies
- Train from scratch → No pre-trained foundation model
- Simpler architecture → Easier to understand and debug
- More established → Proven on many robots

### Architecture Overview

```
Human Demonstration (Teleoperation)
         ↓
    Record:
    ├─ Camera Images (RGB)
    ├─ Robot State (joint positions, velocities)
    └─ Actions (what human commanded)
         ↓
    LeRobot Dataset
         ↓
┌─────────────────────────────────────┐
│       Train Policy Model             │
│   (ACT, Diffusion, TDMPC, etc.)     │
│                                      │
│  Input: Camera + Robot State        │
│  Output: Actions                     │
└─────────────────────────────────────┘
         ↓
    Trained Policy
         ↓
    Deploy on Robot
    ├─ Sees environment (camera)
    ├─ Senses state (joints, IMU)
    └─ Outputs actions (joint commands)
```

### Expected Timeline

| Phase | Duration | Effort | Risk |
|-------|----------|--------|------|
| Installation | 1-2 hours | Low | Low |
| Camera Setup | 1-2 days | Low | Low |
| Data Collection | 3-7 days | Medium | Low |
| Training | 2-6 hours | Low | Low |
| Evaluation | 1-2 days | Medium | Medium |
| Deployment | 1-3 days | Medium | Low |

**Total estimated time**: 1-2 weeks part-time

## What is LeRobot

### Project Overview

LeRobot is **Hugging Face's open-source framework** for robotics AI, launched in 2024:

- Led by ex-Tesla lead **Remi Cadene**
- Focused on making robotics accessible to everyone
- Provides **models, datasets, and tools** in PyTorch
- Integrated with **Hugging Face Hub** for easy sharing
- Emphasizes **imitation learning** and reinforcement learning
- Proven on **real-world robots** (SO-100, ALOHA, etc.)

### Philosophy

**"Show, don't tell"** - Instead of defining rewards (RL) or writing code, you:
1. Demonstrate the task via teleoperation
2. Record demonstrations
3. Train policy to imitate you
4. Deploy to robot

### Major Partnerships (2025)

- **NVIDIA**: Integration with Isaac Sim and Groot N1
- **Pollen Robotics**: Acquisition for humanoid development
- **The Robot Studio**: SO-100 affordable robotic arm ($100)
- **Yaak AI**: Self-driving dataset collaboration (L2D - 1 PB data)

### Supported Robots

**Commercial:**
- ALOHA (Mobile ALOHA)
- SO-100/SO-101 robotic arms
- Koch robots
- XArm manipulators
- Unitree Go2 quadruped

**In Development:**
- HopeJR humanoid ($3,000)
- Reachy Mini humanoid ($250-300)

**Custom robots** (like Berkeley Humanoid Lite) supported via dataset format

## Available Policy Models

### 1. ACT (Action Chunking with Transformers)

**Architecture**: Transformer-based with action chunking

**How it works:**
- Predicts multiple future actions at once (chunk of 10-100 actions)
- Uses CVAE (Conditional VAE) for multimodal action distributions
- Trained on ALOHA datasets for bimanual manipulation

**Best for:**
- Fine manipulation tasks
- Dexterous hand control
- Tasks requiring temporal consistency
- When you have high-quality demonstrations

**Performance:**
- Success rate: 44% on PushT benchmark
- Smooth action trajectories
- Good for precision tasks

**Pros:**
- ✅ Temporally coherent actions
- ✅ Handles multimodal distributions
- ✅ Fast inference (~50 Hz capable)
- ✅ Proven on real robots (ALOHA)

**Cons:**
- ⚠️ Requires more demonstrations (~50-100 episodes)
- ⚠️ Less robust to out-of-distribution states
- ⚠️ Larger model size

**Inference speed**: 20-50 ms per forward pass

### 2. Diffusion Policy

**Architecture**: Conditional diffusion model for action generation

**How it works:**
- Starts with Gaussian noise
- Iteratively denoises to produce actions
- Conditions on observations (camera + state)
- Multiple denoising steps (10-100) to refine actions

**Best for:**
- Complex manipulation
- Tasks with diverse solutions
- When you need high precision
- Multi-step reasoning

**Performance:**
- Success rate: 62-68% on PushT benchmark
- High precision on intricate tasks
- Better generalization than ACT

**Pros:**
- ✅ Best-in-class performance
- ✅ Handles multimodal action distributions well
- ✅ Robust to noise and disturbances
- ✅ State-of-the-art on manipulation benchmarks

**Cons:**
- ⚠️ Slower inference (100-200 ms with 100 steps)
- ⚠️ Can be harder to tune
- ⚠️ Requires more compute

**Inference speed**: 100-200 ms (depends on denoising steps)

### 3. TDMPC (Temporal Difference Model Predictive Control)

**Architecture**: Model-based RL with learned world model

**How it works:**
- Learns dynamics model of environment
- Plans actions using model predictive control
- Uses temporal difference learning

**Best for:**
- Tasks with clear dynamics
- When sample efficiency matters
- Environments where you can simulate

**Pros:**
- ✅ Sample efficient
- ✅ Can plan ahead
- ✅ Works with sparse demonstrations

**Cons:**
- ⚠️ More complex to set up
- ⚠️ Requires environment model
- ⚠️ Less tested on real robots

**Inference speed**: 30-60 ms

### 4. VQ-BeT (Vector-Quantized Behavior Transformer)

**Architecture**: Behavior cloning with discrete action tokens

**How it works:**
- Quantizes continuous actions into discrete tokens
- Uses transformer to predict action sequences
- Similar to language modeling but for actions

**Best for:**
- Long-horizon tasks
- Sequential decision making
- When actions have structure

**Pros:**
- ✅ Good for long sequences
- ✅ Can leverage language model techniques
- ✅ Handles temporal dependencies well

**Cons:**
- ⚠️ Action quantization can lose precision
- ⚠️ Newer, less battle-tested
- ⚠️ Requires tuning codebook size

**Inference speed**: 20-40 ms

### Model Recommendations for Berkeley Humanoid Lite

| Task Type | Best Model | Second Choice | Why |
|-----------|-----------|---------------|-----|
| **Locomotion** | Diffusion Policy | ACT | Needs robust control, precision |
| **Arm gestures** | ACT | Diffusion Policy | Smooth trajectories important |
| **Manipulation** | Diffusion Policy | ACT | High precision required |
| **Whole-body coordination** | Diffusion Policy | TDMPC | Complex multi-limb control |
| **Fast deployment** | ACT | VQ-BeT | Faster inference needed |

**Default recommendation**: Start with **Diffusion Policy** for best performance, switch to **ACT** if you need faster inference.

## Prerequisites

### Hardware Requirements

**Required:**
- Development workstation with NVIDIA GPU (RTX 3060+, 8GB+ VRAM)
- Berkeley Humanoid Lite robot
- RGB camera (same options as Groot N1)
- Teleoperation interface (gamepad, VR, or keyboard)

**Optional:**
- Jetson Orin Nano/AGX for onboard inference
- Multiple cameras for better perception
- Motion capture system for ground truth

### Software Requirements

**Development Machine:**
- Ubuntu 20.04/22.04
- Python 3.10 or 3.11
- PyTorch 2.2+
- CUDA 11.8+ (if using GPU)
- FFmpeg with libsvtav1 encoder

**Robot (for deployment):**
- Same as current setup (N95 or Jetson)
- Camera drivers
- Network connection (WiFi/Ethernet)

### Knowledge Prerequisites

- Basic Python
- Familiarity with PyTorch
- Robot teleoperation experience
- Understanding of imitation learning (helpful but not required)

## Installation

### Step 1: Create Environment

```bash
# On development workstation
cd ~/Projects

# Create dedicated environment
python3.10 -m venv ~/lerobot_env
source ~/lerobot_env/bin/activate

# Or use conda
conda create -n lerobot python=3.10
conda activate lerobot
```

### Step 2: Install LeRobot

```bash
# Install from PyPI (recommended for stable use)
pip install lerobot

# Or install from source (for latest features)
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .

# Install all extras (simulation environments, optional)
pip install 'lerobot[all]'
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    opencv-python \
    gymnasium \
    imageio \
    imageio-ffmpeg \
    av \
    pyserial

# For visualization
pip install matplotlib pillow

# For Hugging Face integration
pip install huggingface_hub datasets

# Optional: Weights & Biases for tracking
pip install wandb
```

### Step 4: Install FFmpeg with AV1 Support

```bash
# Ubuntu
sudo apt update
sudo apt install ffmpeg libavcodec-extra

# Verify
ffmpeg -codecs | grep av1
# Should show: libsvtav1 or libaom-av1
```

### Step 5: Verify Installation

```bash
# Test import
python << 'EOF'
import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

print(f"✓ LeRobot version: {lerobot.__version__}")
print("✓ Policies imported successfully")
EOF

# Test training CLI
lerobot-train --help
```

### Step 6: Login to Hugging Face (Optional)

```bash
# For uploading datasets/models
pip install huggingface_hub
huggingface-cli login

# Enter token from https://huggingface.co/settings/tokens
```

## Understanding LeRobot Dataset Format

### Dataset Structure

LeRobot datasets are stored in a unified format:

```
dataset_name/
├── meta_data/
│   ├── info.json              # Dataset metadata
│   ├── stats.json             # Normalization statistics
│   └── episodes.json          # Episode indices
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet    # Observations, actions
│   │   ├── observation.images.cam_0_*.mp4  # Video frames
│   │   └── ...
│   ├── chunk-001/
│   └── ...
└── videos/                    # Replay videos (optional)
```

### Key Files

**info.json**: Dataset configuration
```json
{
  "fps": 30,
  "robot_type": "berkeley_humanoid_lite",
  "keys": {
    "observation.images.head_camera": {"shape": [480, 640, 3], "dtype": "video"},
    "observation.state": {"shape": [22], "dtype": "float32"},
    "action": {"shape": [22], "dtype": "float32"}
  }
}
```

**stats.json**: Normalization statistics
```json
{
  "observation.state": {
    "mean": [0.0, 0.1, ...],
    "std": [0.5, 0.3, ...]
  },
  "action": {
    "mean": [0.0, 0.0, ...],
    "std": [0.2, 0.2, ...]
  }
}
```

**episodes.json**: Episode metadata
```json
{
  "0": {"length": 243, "timestamp": "2025-10-23T12:34:56"},
  "1": {"length": 189, "timestamp": "2025-10-23T12:45:12"},
  ...
}
```

### Data Format

Each episode's parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| `frame_index` | int | Frame number within episode |
| `timestamp` | float | Time since episode start (seconds) |
| `episode_index` | int | Which episode this frame belongs to |
| `observation.images.head_camera` | path | Path to video frame |
| `observation.state` | float32[22] | Joint positions |
| `observation.velocity` | float32[22] | Joint velocities (optional) |
| `action` | float32[22] | Joint position commands |
| `next.done` | bool | Episode termination flag |
| `next.success` | bool | Task success flag (optional) |
| `next.reward` | float | Reward (optional, for RL) |

## Phase 1: Camera Setup

### Hardware Installation

Same as Groot N1 integration - see that document for:
- Camera mounting options (fixed vs 1-DOF neck)
- Hardware recommendations (Pi Camera, Logitech C920)
- Physical integration

### Software Setup

```bash
# Install camera interface
pip install opencv-python

# Create camera module for Berkeley Humanoid Lite
mkdir -p ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel
```

Create camera interface:
```python
# File: source/berkeley_humanoid_lite_lowlevel/camera_interface.py

import cv2
import numpy as np
from typing import Tuple, Optional
import threading
import queue

class CameraInterface:
    """Non-blocking camera interface for LeRobot"""

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        buffer_size: int = 2
    ):
        self.resolution = resolution
        self.fps = fps

        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        # Frame buffer
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None

        print(f"✓ Camera {camera_id} opened: {resolution[0]}x{resolution[1]} @ {fps} FPS")

    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Background capture thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Non-blocking put
                try:
                    self.frame_queue.put(frame_rgb, block=False)
                except queue.Full:
                    # Drop old frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame_rgb, block=False)
                    except queue.Empty:
                        pass

    def read(self) -> Optional[np.ndarray]:
        """Get latest frame (non-blocking)"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self):
        """Stop capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
```

Test camera:
```bash
python << 'EOF'
from berkeley_humanoid_lite_lowlevel.camera_interface import CameraInterface
import time

with CameraInterface(camera_id=0) as camera:
    for i in range(30):
        frame = camera.read()
        if frame is not None:
            print(f"Frame {i}: {frame.shape}, dtype={frame.dtype}")
        time.sleep(0.033)  # 30 FPS

print("✓ Camera test successful")
EOF
```

## Phase 2: Data Collection

### Setup Recording Environment

```python
# Create data recording script
# File: scripts/lerobot/record_dataset.py

import argparse
import time
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from berkeley_humanoid_lite_lowlevel.camera_interface import CameraInterface
# Import your robot control interface

def record_episode(
    robot,
    camera: CameraInterface,
    dataset: LeRobotDataset,
    episode_index: int,
    max_duration: float = 30.0
) -> int:
    """Record one teleoperation episode"""

    print(f"\n=== Recording Episode {episode_index} ===")
    print("Press SPACE to start, ESC to finish episode")

    # Wait for start signal
    input("Press ENTER when ready...")

    frames = []
    start_time = time.time()
    frame_index = 0

    try:
        while time.time() - start_time < max_duration:
            # Get observations
            image = camera.read()
            if image is None:
                continue

            state = robot.get_joint_positions()  # (22,)
            velocity = robot.get_joint_velocities()  # (22,)

            # Get action (what user commanded via teleoperation)
            action = robot.get_commanded_positions()  # (22,)

            # Store frame
            frame_data = {
                'timestamp': time.time() - start_time,
                'frame_index': frame_index,
                'episode_index': episode_index,
                'observation.images.head_camera': image,
                'observation.state': state,
                'observation.velocity': velocity,
                'action': action,
                'next.done': False,  # Will set True for last frame
            }
            frames.append(frame_data)
            frame_index += 1

            # Control rate
            time.sleep(1/30)  # 30 Hz

    except KeyboardInterrupt:
        print("\nEpisode stopped by user")

    # Mark last frame as done
    if frames:
        frames[-1]['next.done'] = True

    # Add to dataset
    dataset.add_episode(frames)

    print(f"✓ Episode {episode_index} complete: {len(frames)} frames")
    return len(frames)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--camera-id', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='data/lerobot')
    args = parser.parse_args()

    # Initialize camera
    camera = CameraInterface(camera_id=args.camera_id)
    camera.start()

    # Initialize robot (placeholder - use your actual interface)
    # robot = RobotInterface()

    # Create dataset
    dataset_path = Path(args.output_dir) / args.dataset_name
    dataset = LeRobotDataset.create(
        repo_id=args.dataset_name,
        root=str(dataset_path),
        robot_type="berkeley_humanoid_lite",
        fps=30,
        video=True,  # Save as compressed video
        keys=[
            "observation.images.head_camera",
            "observation.state",
            "observation.velocity",
            "action"
        ]
    )

    # Record episodes
    total_frames = 0
    for i in range(args.num_episodes):
        num_frames = record_episode(robot, camera, dataset, i)
        total_frames += num_frames

        # Short break between episodes
        print(f"\nCompleted {i+1}/{args.num_episodes} episodes ({total_frames} total frames)")
        time.sleep(2)

    # Save dataset
    dataset.save()
    print(f"\n✓ Dataset saved to {dataset_path}")
    print(f"✓ Total: {args.num_episodes} episodes, {total_frames} frames")

    camera.stop()

if __name__ == "__main__":
    main()
```

### Recording Workflow

```bash
# Terminal 1: Start teleoperation interface
# (Use your existing teleop code)
cd ~/Projects/Berkeley-Humanoid-Lite
python scripts/teleop/run_teleop.py

# Terminal 2: Record demonstrations
python scripts/lerobot/record_dataset.py \
    --dataset-name berkeley_humanoid_lite_walking \
    --num-episodes 50 \
    --camera-id 0

# For each episode:
# 1. Position robot in starting pose
# 2. Press ENTER to start recording
# 3. Teleoperate robot to complete task
# 4. Press ESC to finish episode
# 5. Repeat
```

### Data Collection Guidelines

**Quantity:**
- Minimum: 20-30 episodes
- Recommended: 50-100 episodes
- More data = better performance

**Quality over quantity:**
- Smooth, natural movements
- Successful task completion
- Consistent demonstrations
- Vary starting conditions slightly

**Diversity:**
- Include variations (speed, path, orientation)
- Show recovery behaviors
- Cover edge cases
- Multiple lighting conditions

**Tips:**
- Practice teleoperation first
- Record in batches (10-20 episodes at a time)
- Review recordings before continuing
- Delete failed episodes

### Push to Hugging Face Hub (Optional)

```bash
# Upload dataset for sharing
python << 'EOF'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "berkeley_humanoid_lite_walking",
    root="data/lerobot/berkeley_humanoid_lite_walking"
)

# Push to Hub
dataset.push_to_hub(
    repo_id="your_username/berkeley_humanoid_lite_walking",
    private=False  # or True for private
)
EOF
```

## Phase 3: Training

### Configure Training

```yaml
# Create training config
# File: scripts/lerobot/configs/diffusion_humanoid.yaml

# Experiment settings
seed: 42
device: cuda
use_amp: true  # Mixed precision training

# Dataset
dataset_repo_id: data/lerobot/berkeley_humanoid_lite_walking
training_ratio: 0.95
val_ratio: 0.05

# Policy
policy:
  name: diffusion
  n_obs_steps: 2  # Number of observation timesteps to stack
  n_action_steps: 8  # Number of actions to predict
  horizon: 16  # Total action horizon
  n_action_repeats: 1  # How many times to repeat each action

  # Vision encoder
  vision_backbone: resnet18
  pretrained_backbone: true

  # State encoder
  input_shapes:
    observation.state: [22]
    observation.velocity: [22]
    observation.images.head_camera: [3, 480, 640]
  output_shapes:
    action: [22]

  # Diffusion settings
  num_inference_steps: 10  # Fewer = faster, more = better
  down_dims: [256, 512, 1024]

# Training
training:
  batch_size: 16
  num_epochs: 500
  grad_clip_norm: 10.0
  lr: 1e-4
  lr_scheduler: cosine
  warmup_steps: 500
  weight_decay: 1e-6

  # Data augmentation
  image_augmentation:
    enable: true
    random_crop: [0.9, 1.0]
    brightness: 0.1
    contrast: 0.1

  # Logging
  log_freq: 10
  eval_freq: 100
  save_freq: 500

  # Checkpointing
  save_checkpoint: true
  checkpoint_dir: outputs/diffusion_humanoid

# Evaluation
eval:
  n_episodes: 10
  batch_size: 1
```

### Start Training

```bash
# Using config file
cd ~/Projects/Berkeley-Humanoid-Lite

python -m lerobot.scripts.train \
    --config scripts/lerobot/configs/diffusion_humanoid.yaml \
    --output-dir outputs/diffusion_humanoid_walking

# Or using command line args
python -m lerobot.scripts.train \
    policy=diffusion \
    dataset_repo_id=data/lerobot/berkeley_humanoid_lite_walking \
    training.batch_size=16 \
    training.num_epochs=500 \
    device=cuda \
    output_dir=outputs/diffusion_humanoid_walking

# With Weights & Biases tracking
wandb login
python -m lerobot.scripts.train \
    --config scripts/lerobot/configs/diffusion_humanoid.yaml \
    --wandb-project berkeley-humanoid-lite \
    --wandb-run-name diffusion-walking-v1
```

### Training Time Estimates

| Dataset Size | Model | GPU | Time |
|--------------|-------|-----|------|
| 30 episodes (~5K frames) | Diffusion | RTX 3090 | 2-3 hours |
| 30 episodes | ACT | RTX 3090 | 1-2 hours |
| 100 episodes (~17K frames) | Diffusion | RTX 3090 | 4-6 hours |
| 100 episodes | ACT | RTX 3090 | 3-4 hours |
| 30 episodes | Diffusion | RTX PRO 6000 | 1-2 hours |

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir outputs/diffusion_humanoid_walking

# Open http://localhost:6006

# Watch for:
# - Training loss (should decrease)
# - Validation loss (should decrease, not overfit)
# - Action prediction error
# - Learning rate schedule
```

### Train Different Models

```bash
# ACT policy (faster inference)
python -m lerobot.scripts.train \
    policy=act \
    dataset_repo_id=data/lerobot/berkeley_humanoid_lite_walking \
    training.batch_size=8 \
    output_dir=outputs/act_humanoid_walking

# TDMPC (model-based)
python -m lerobot.scripts.train \
    policy=tdmpc \
    dataset_repo_id=data/lerobot/berkeley_humanoid_lite_walking \
    output_dir=outputs/tdmpc_humanoid_walking

# VQ-BeT
python -m lerobot.scripts.train \
    policy=vqbet \
    dataset_repo_id=data/lerobot/berkeley_humanoid_lite_walking \
    output_dir=outputs/vqbet_humanoid_walking
```

## Phase 4: Evaluation

### Evaluate in Simulation (If Available)

```python
# Evaluate trained policy
# File: scripts/lerobot/evaluate_policy.py

import torch
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def evaluate_policy(
    checkpoint_path: str,
    dataset_path: str,
    num_episodes: int = 10
):
    """Evaluate trained policy on validation set"""

    # Load policy
    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to('cuda')

    # Load dataset
    dataset = LeRobotDataset(dataset_path, split='val')

    # Evaluation metrics
    total_mse = 0
    total_frames = 0

    with torch.no_grad():
        for episode_idx in range(num_episodes):
            episode = dataset.get_episode(episode_idx)

            for frame in episode:
                # Get observation
                obs = {
                    'observation.images.head_camera': frame['observation.images.head_camera'],
                    'observation.state': frame['observation.state'],
                }

                # Predict action
                predicted_action = policy(obs)

                # Ground truth action
                true_action = frame['action']

                # Compute MSE
                mse = torch.mean((predicted_action - true_action) ** 2)
                total_mse += mse.item()
                total_frames += 1

    # Average MSE
    avg_mse = total_mse / total_frames
    print(f"Average MSE: {avg_mse:.6f}")
    return avg_mse

# Run evaluation
evaluate_policy(
    checkpoint_path="outputs/diffusion_humanoid_walking/checkpoints/last",
    dataset_path="data/lerobot/berkeley_humanoid_lite_walking",
    num_episodes=10
)
```

### Visualize Predictions

```python
# Visualize policy predictions
# File: scripts/lerobot/visualize_policy.py

import numpy as np
import matplotlib.pyplot as plt
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def visualize_predictions(checkpoint_path, dataset_path, episode_idx=0):
    """Visualize predicted vs actual actions"""

    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to('cuda')

    dataset = LeRobotDataset(dataset_path, split='val')
    episode = dataset.get_episode(episode_idx)

    predictions = []
    ground_truth = []

    for frame in episode:
        obs = {
            'observation.images.head_camera': frame['observation.images.head_camera'],
            'observation.state': frame['observation.state'],
        }

        pred = policy(obs).cpu().numpy()
        gt = frame['action'].numpy()

        predictions.append(pred)
        ground_truth.append(gt)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Plot
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    for i, ax in enumerate(axes.flat):
        if i >= 22:  # Berkeley Humanoid Lite has 22 DOF
            break
        ax.plot(ground_truth[:, i], label='Ground Truth', alpha=0.7)
        ax.plot(predictions[:, i], label='Predicted', alpha=0.7)
        ax.set_title(f'Joint {i}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('policy_visualization.png')
    print("✓ Saved visualization to policy_visualization.png")

visualize_predictions(
    "outputs/diffusion_humanoid_walking/checkpoints/last",
    "data/lerobot/berkeley_humanoid_lite_walking",
    episode_idx=0
)
```

## Phase 5: Deployment

### Deploy to Real Robot

```python
# Real-time policy deployment
# File: scripts/lerobot/deploy_realtime.py

import torch
import time
import numpy as np
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from berkeley_humanoid_lite_lowlevel.camera_interface import CameraInterface
# Import your robot interface

class PolicyController:
    """Real-time policy controller"""

    def __init__(
        self,
        checkpoint_path: str,
        camera_id: int = 0,
        control_freq: float = 30.0  # Hz
    ):
        # Load policy
        print("Loading policy...")
        self.policy = DiffusionPolicy.from_pretrained(checkpoint_path)
        self.policy.eval()
        self.policy.to('cuda')
        print("✓ Policy loaded")

        # Initialize camera
        self.camera = CameraInterface(camera_id=camera_id, fps=30)
        self.camera.start()
        print("✓ Camera started")

        # Initialize robot
        # self.robot = RobotInterface()
        # print("✓ Robot connected")

        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

    def run(self, duration: float = 30.0):
        """Run policy for specified duration"""

        print(f"\nRunning policy for {duration} seconds...")
        print("Press Ctrl+C to stop\n")

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                loop_start = time.time()

                # Get observation
                image = self.camera.read()
                if image is None:
                    continue

                # state = self.robot.get_joint_positions()

                # Prepare input
                obs = {
                    'observation.images.head_camera': torch.from_numpy(image).cuda().float() / 255.0,
                    # 'observation.state': torch.from_numpy(state).cuda().float(),
                }

                # Policy inference
                with torch.no_grad():
                    action = self.policy(obs)

                # Send to robot
                action_np = action.cpu().numpy()[0]  # (22,)
                # self.robot.set_joint_commands(action_np)

                # Timing
                frame_count += 1
                elapsed = time.time() - loop_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)

                # Print stats
                if frame_count % 30 == 0:
                    actual_freq = 1.0 / (time.time() - loop_start)
                    print(f"Frame {frame_count}: {actual_freq:.1f} Hz")

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            self.camera.stop()
            # self.robot.stop()

        elapsed = time.time() - start_time
        print(f"\n✓ Completed {frame_count} frames in {elapsed:.1f}s")
        print(f"✓ Average frequency: {frame_count/elapsed:.1f} Hz")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--camera-id', type=int, default=0)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--freq', type=float, default=30.0)
    args = parser.parse_args()

    controller = PolicyController(
        checkpoint_path=args.checkpoint,
        camera_id=args.camera_id,
        control_freq=args.freq
    )

    controller.run(duration=args.duration)

if __name__ == "__main__":
    main()
```

### Run Deployment

```bash
# Deploy trained policy
python scripts/lerobot/deploy_realtime.py \
    --checkpoint outputs/diffusion_humanoid_walking/checkpoints/last \
    --camera-id 0 \
    --duration 60 \
    --freq 30
```

### Safety Considerations

**Before running on robot:**
1. Test in simulation first (if possible)
2. Start with low PD gains
3. Have emergency stop ready
4. Test with robot on stand (legs off ground)
5. Gradually increase confidence
6. Monitor for unexpected behaviors

## Model Comparison Guide

### When to Use Each Model

**Diffusion Policy:**
- ✅ Use when performance is critical
- ✅ Use for complex manipulation
- ✅ Use when you have GPU compute
- ❌ Avoid if you need <50ms inference
- ❌ Avoid on CPU-only systems

**ACT:**
- ✅ Use for smooth trajectories
- ✅ Use when inference speed matters
- ✅ Use for bimanual coordination
- ❌ Avoid for highly multimodal tasks

**TDMPC:**
- ✅ Use when data is limited
- ✅ Use for sample-efficient learning
- ❌ Avoid if environment model is complex
- ❌ Avoid for real-time requirements

### Performance Comparison

| Model | Success Rate | Inference | Memory | Sample Efficiency |
|-------|--------------|-----------|--------|-------------------|
| Diffusion | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| ACT | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| TDMPC | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| VQ-BeT | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

## Performance Optimization

### Speed Up Inference

```python
# Use TorchScript for faster inference
import torch

policy = DiffusionPolicy.from_pretrained(checkpoint_path)
policy.eval()

# Compile model
scripted_policy = torch.jit.script(policy)
scripted_policy.save("policy_scripted.pt")

# Load and use
fast_policy = torch.jit.load("policy_scripted.pt")
fast_policy.to('cuda')
```

### Reduce Model Size

```python
# Quantize model to FP16
policy = policy.half()  # Convert to FP16

# Or use INT8 quantization
policy = torch.quantization.quantize_dynamic(
    policy, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Optimize Diffusion Steps

```python
# Fewer diffusion steps = faster inference
# Default: 100 steps
# Fast: 10 steps (10x faster, slight quality drop)

policy.num_inference_steps = 10
```

## Troubleshooting

### Issue: Policy doesn't generalize

**Symptoms**: Works on training data, fails on robot

**Solutions:**
- Collect more diverse demonstrations
- Add data augmentation (random crops, color jitter)
- Check observation normalization
- Verify camera calibration matches training
- Test in similar lighting conditions

### Issue: Actions are jerky/unstable

**Symptoms**: Robot movements are shaky

**Solutions:**
- Use ACT instead of Diffusion (smoother)
- Increase `n_action_steps` (longer horizon)
- Add action smoothing filter
- Lower PD gains
- Check control frequency

### Issue: Slow inference

**Symptoms**: Can't achieve target frequency

**Solutions:**
- Use ACT or VQ-BeT (faster models)
- Reduce diffusion steps (10 instead of 100)
- Use TorchScript compilation
- Reduce image resolution
- Use FP16 precision

### Issue: High memory usage

**Symptoms**: Out of memory errors

**Solutions:**
- Use smaller vision backbone (resnet18 → mobilenet)
- Reduce batch size
- Use gradient checkpointing
- Clear CUDA cache
- Use FP16 training

## LeRobot vs RL vs Groot N1

### Comparison Matrix

| Aspect | RL (RSL-RL) | LeRobot | Groot N1 |
|--------|-------------|---------|----------|
| **Learning paradigm** | Reinforcement Learning | Imitation Learning | Foundation Model |
| **Data source** | Simulation | Human demos | Pre-trained + fine-tune |
| **Training time** | 2 hours | 2-6 hours | 1-2 days (fine-tune) |
| **Data collection** | Automatic | Manual (teleoperation) | Manual (demos) |
| **Reward engineering** | Required | Not needed | Not needed |
| **Vision** | Optional | Required | Required |
| **Language** | No | No | Yes |
| **Inference speed** | 25 Hz | 10-50 Hz | 10 Hz |
| **Model size** | ~10 MB | ~50-200 MB | ~5 GB |
| **Compute (inference)** | CPU ok | GPU preferred | GPU required |
| **Sim2real** | Excellent | Good | Unknown |
| **Multi-task** | No (one policy per task) | No (one policy per task) | Yes |
| **Generalization** | Task-specific | Task-specific | Broad |
| **Ease of use** | Medium | Easy | Medium |
| **Debugging** | Hard (reward tuning) | Easy (check demos) | Medium |

### When to Use Each

**Use RL when:**
- You have good simulator
- You can define reward function
- You want maximum performance
- You don't want to collect demonstrations
- Task is locomotion or navigation

**Use LeRobot when:**
- Hard to define reward function
- You can demonstrate the task
- You need vision-based control
- You want fast iteration
- Task is manipulation or dexterous

**Use Groot N1 when:**
- You want language conditioning
- You need multi-task capability
- You can fine-tune foundation model
- You have powerful GPU
- You want cutting-edge research

### Hybrid Approach

Best of all worlds:

1. **RL for locomotion** → Fast, efficient, proven
2. **LeRobot for manipulation** → Easy to teach, vision-based
3. **Groot N1 for coordination** → Language commands, multi-task

Example workflow:
```
User: "Walk to the table and wave"
  ↓
Use RL policy for walking (fast, robust)
  ↓
Use LeRobot policy for arm waving (taught via demo)
```

## Next Steps

### Immediate (This Week)
1. Install LeRobot framework
2. Add camera to robot
3. Test teleoperation and recording
4. Collect 10 demo episodes
5. Train first model (start with ACT for speed)

### Short Term (2-4 weeks)
6. Collect 50-100 quality demonstrations
7. Train Diffusion Policy
8. Evaluate on real robot
9. Iterate on data collection
10. Optimize for deployment

### Medium Term (1-3 months)
11. Collect datasets for multiple tasks
12. Compare ACT vs Diffusion vs TDMPC
13. Push datasets to Hugging Face Hub
14. Share results with community

### Long Term
15. Combine with RL locomotion policies
16. Explore multi-task learning
17. Contribute improvements to LeRobot
18. Publish research results

## Resources

- **LeRobot GitHub**: https://github.com/huggingface/lerobot
- **LeRobot Hub**: https://huggingface.co/lerobot
- **Documentation**: https://huggingface.co/docs/lerobot
- **Discord**: Hugging Face Discord #lerobot channel
- **Paper (ACT)**: https://arxiv.org/abs/2304.13705
- **Paper (Diffusion Policy)**: https://arxiv.org/abs/2303.04137
- **SO-100 Tutorial**: https://wiki.seeedstudio.com/lerobot_so100m/

## Conclusion

LeRobot provides an **accessible, powerful framework** for teaching robots through demonstration. Key advantages:

✅ **No reward engineering** - Just show the robot what to do
✅ **Vision-based** - Robot can see and react
✅ **Fast iteration** - Hours of training, not days
✅ **Proven** - Used on many real robots
✅ **Community** - Hugging Face ecosystem support

**Start simple:**
1. Install LeRobot
2. Collect 20-30 demonstrations
3. Train ACT policy (fastest)
4. Test on robot
5. Iterate

**Compared to alternatives:**
- **vs RL**: Easier to get started, no reward tuning
- **vs Groot N1**: Faster training, less compute, simpler
- **Trade-off**: Task-specific (not multi-task like Groot N1)

LeRobot is ideal for **Berkeley Humanoid Lite manipulation tasks** where you can demonstrate the desired behavior. Combine with RL for locomotion for a complete system!
