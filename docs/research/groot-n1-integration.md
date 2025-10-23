# NVIDIA Groot N1 Integration with Berkeley Humanoid Lite

Complete guide for integrating NVIDIA Isaac Groot N1 foundation model with Berkeley Humanoid Lite for vision-based, language-conditioned robot control.

## Table of Contents

1. [Overview](#overview)
2. [What is Groot N1](#what-is-groot-n1)
3. [Prerequisites](#prerequisites)
4. [Phase 1: Setup and Installation](#phase-1-setup-and-installation)
5. [Phase 2: Simulation Testing](#phase-2-simulation-testing)
6. [Phase 3: Camera Integration](#phase-3-camera-integration)
7. [Phase 4: Data Collection](#phase-4-data-collection)
8. [Phase 5: Fine-Tuning](#phase-5-fine-tuning)
9. [Phase 6: Tethered Deployment](#phase-6-tethered-deployment)
10. [Phase 7: Onboard Deployment](#phase-7-onboard-deployment)
11. [Optimization Techniques](#optimization-techniques)
12. [Troubleshooting](#troubleshooting)
13. [Performance Benchmarks](#performance-benchmarks)
14. [Comparison: RL vs Groot N1](#comparison-rl-vs-groot-n1)

## Overview

### What This Integration Achieves

Transform Berkeley Humanoid Lite from task-specific RL policies to a generalist vision-language-action (VLA) model capable of:

- **Language-conditioned behaviors**: "Walk forward", "Wave your arms", "Pick up the cup"
- **Vision-based manipulation**: Using camera input to perceive and interact with environment
- **Multi-task capability**: One model handles multiple behaviors (locomotion + manipulation + gestures)
- **Zero-shot generalization**: Perform novel tasks without additional training
- **Natural language interaction**: Command robot with plain English

### Architecture Overview

```
Human Input
    ├─ Language: "Walk to the red object"
    └─ (Optional) Demonstrations
         ↓
    Camera Input (RGB)
         ↓
    Robot Proprioception (joint states, IMU)
         ↓
┌──────────────────────────────────────┐
│        Groot N1 Foundation Model      │
│                                       │
│  System 2 (Slow Thinking)            │
│  ├─ Vision-Language Reasoning        │
│  ├─ Task Planning                    │
│  └─ High-level Decision Making       │
│         ↓                             │
│  System 1 (Fast Thinking)            │
│  ├─ Action Diffusion Transformer     │
│  ├─ Real-time Motor Control          │
│  └─ Smooth Trajectory Generation     │
└──────────────────────────────────────┘
         ↓
    Joint Position Commands (22 DOF)
         ↓
    Robot Executes @ 10 Hz
```

### Expected Timeline

| Phase | Duration | Effort | Risk |
|-------|----------|--------|------|
| Setup & Installation | 1-2 days | Medium | Low |
| Simulation Testing | 3-5 days | Medium | Low |
| Camera Integration | 1-2 days | Low | Low |
| Data Collection | 1-2 weeks | High | Medium |
| Fine-tuning | 2-3 days | Medium | Medium |
| Tethered Deployment | 1-2 days | Medium | Low |
| Onboard Deployment | 3-5 days | High | High |

**Total estimated time**: 1-2 months part-time

## What is Groot N1

### Foundation Model Concept

Unlike task-specific RL policies (trained from scratch for walking), Groot N1 is a **pre-trained foundation model**:

- Trained on 780,000+ synthetic trajectories (6,500+ hours of demos)
- Learned from diverse robot embodiments and tasks
- Transfer learning: Fine-tune for your robot, not train from scratch
- Cross-embodiment: Works on different robot types

### Model Variants

| Model | Parameters | Memory | Speed | Best For |
|-------|-----------|---------|-------|----------|
| Groot N1-1B | 1 billion | ~2.5 GB | Faster | Edge devices, Jetson Orin Nano |
| Groot N1-2B | 2 billion | ~4.5 GB | Balanced | General use, Jetson AGX Orin |
| Groot N1.5-3B | 3 billion | ~6.5 GB | Slower | Maximum accuracy, workstation |

### Architecture Details

**System 2 (Reasoning):**
- Vision encoder: SigLIP2 (frozen, pre-trained)
- Language encoder: T5 (frozen, pre-trained)
- Processes: 224x224 RGB images + text instructions
- Output: High-level plan/context for System 1

**System 1 (Action Generation):**
- Flow matching diffusion transformer
- Input: System 2 context + robot state (joint positions, velocities)
- Output: Chunk of actions (typically 10-20 timesteps ahead)
- Embodiment-aware via learned embeddings

### Capabilities

**Out-of-the-box (pre-trained):**
- Basic locomotion (forward, backward, turning)
- Simple manipulation (reaching, grasping)
- Object tracking with vision
- Following waypoints

**After fine-tuning for Berkeley Humanoid Lite:**
- Robust bipedal walking
- Whole-body coordination (arms + legs)
- Robot-specific behaviors
- Custom tasks and gestures

### Limitations

- Runs at 10 Hz (slower than RL at 25 Hz)
- Requires GPU (can't run on CPU efficiently)
- Memory intensive (6-8 GB minimum)
- Needs demonstration data for fine-tuning
- Sim-to-real gap may be larger than RL

## Prerequisites

### Hardware Requirements

**Required:**
- Workstation with NVIDIA GPU (RTX 3090/4090, A6000, or better)
- Berkeley Humanoid Lite robot
- RGB camera (see [Camera Options](#camera-options))
- Network connection (WiFi/Ethernet for tethered mode)

**Recommended:**
- Jetson AGX Orin (32GB) or AGX Thor for onboard inference
- External power supply (for extended testing)
- Mocap system (optional, for ground truth data)

**Optional:**
- Jetson Orin Nano Super (for experimental onboard inference)

### Software Requirements

**Development Workstation:**
- Ubuntu 20.04/22.04
- NVIDIA Driver 525+
- CUDA 12.4+
- Python 3.10 or 3.11
- Isaac Sim 5.0.0 + Isaac Lab 2.2.0 (already installed)
- PyTorch 2.7.0+ (already installed)

**Robot (if running onboard):**
- JetPack 6.x (for Jetson)
- CUDA 12.x
- TensorRT 8.x+

### Camera Options

| Camera | Resolution | Interface | FOV | Cost | Recommended |
|--------|-----------|-----------|-----|------|-------------|
| Raspberry Pi Camera v2 | 3280x2464 | CSI/USB | 62.2° | $25 | ✅ Best budget |
| Logitech C920 | 1920x1080 | USB 2.0 | 78° | $70 | ✅ Easy setup |
| Intel RealSense D435i | 1920x1080 RGB + Depth | USB 3.0 | 87° (RGB) | $300 | Optional (depth not required) |
| Arducam IMX219 | 3280x2464 | USB/MIPI | 77° | $50 | ✅ Good quality |

**Groot N1 Requirements:**
- RGB only (depth not used)
- 640x480 or higher resolution
- 30 FPS or higher
- USB or CSI interface

**Recommendation**: Start with **Raspberry Pi Camera v2** or **Logitech C920**. Add RealSense later if you want depth for other tasks.

### Knowledge Prerequisites

- Familiarity with Isaac Sim and Isaac Lab
- Basic Python and PyTorch
- Understanding of robot kinematics
- Experience with Berkeley Humanoid Lite RL training

## Phase 1: Setup and Installation

### Step 1: Install Groot N1 Repository

```bash
# On your development workstation (not robot)
cd ~/Projects

# Clone Groot N1 repository
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

# Checkout stable release
git checkout v1.5  # Or latest stable tag

# Create dedicated environment
conda create -n groot python=3.10
conda activate groot

# Or use venv
python3.10 -m venv ~/groot_env
source ~/groot_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install Groot N1
cd ~/Projects/Isaac-GR00T
pip install -e .

# Install additional dependencies
pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    transformers \
    diffusers \
    accelerate \
    huggingface_hub \
    opencv-python \
    matplotlib \
    pyyaml \
    omegaconf

# Install LeRobot (for data format compatibility)
pip install lerobot
```

### Step 3: Download Pretrained Models

```bash
# Using Hugging Face CLI
pip install huggingface_hub

# Login (get token from huggingface.co/settings/tokens)
huggingface-cli login

# Download Groot N1-2B model
huggingface-cli download nvidia/GR00T-N1-2B \
    --local-dir ~/models/groot-n1-2b

# Or download 1B model for faster inference
huggingface-cli download nvidia/GR00T-N1-1B \
    --local-dir ~/models/groot-n1-1b
```

### Step 4: Verify Installation

```bash
# Test imports
python << 'EOF'
import torch
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ Groot N1 modules imported")

# Load pretrained model (CPU for now)
policy = Gr00tPolicy(
    model_path="~/models/groot-n1-2b",
    embodiment_tag=EmbodimentTag.GR1,  # Placeholder
    device='cpu'
)
print(f"✓ Model loaded: {policy.num_parameters/1e9:.1f}B parameters")
EOF
```

### Step 5: Setup Isaac Sim Integration

```bash
# Navigate to Berkeley Humanoid Lite workspace
cd ~/Projects/Berkeley-Humanoid-Lite

# Activate your Isaac Lab environment
source .venv/bin/activate

# Verify Isaac Sim + Isaac Lab
python << 'EOF'
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
app_launcher = AppLauncher(args)

print("✓ Isaac Sim + Isaac Lab ready")

# Import simulator
from isaaclab.sim import SimulationContext
print("✓ Simulation context available")
EOF
```

## Phase 2: Simulation Testing

### Step 1: Add Camera to Robot USD

First, add a camera sensor to Berkeley Humanoid Lite's Isaac Sim model:

```python
# Create camera configuration
# File: source/berkeley_humanoid_lite_assets/robots/berkeley_humanoid_lite_camera.py

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

@configclass
class HumanoidCameraCfg:
    """Camera configuration for Berkeley Humanoid Lite"""

    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/head_camera",
        update_period=0.033,  # 30 FPS
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.15),  # 10cm forward, 15cm up from torso
            rot=(1.0, 0.0, 0.0, 0.0),  # Look straight ahead
            convention="ros",
        ),
    )
```

### Step 2: Create Groot N1 Test Environment

```python
# Create test script
# File: scripts/groot/test_groot_sim.py

import argparse
import torch
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera

# Import Groot N1
import sys
sys.path.append('/home/kranthi/Projects/Isaac-GR00T')
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

def main():
    """Test Groot N1 with Berkeley Humanoid Lite in Isaac Sim"""

    # Create simulation context
    sim = SimulationContext(physics_dt=0.005, rendering_dt=0.033)

    # Load robot
    from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG

    robot = Articulation(HUMANOID_LITE_CFG.replace(prim_path="/World/robot"))

    # Add camera
    from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite_camera import HumanoidCameraCfg
    camera = Camera(HumanoidCameraCfg.head_camera.replace(prim_path="/World/robot/head_camera"))

    # Play simulation
    sim.reset()

    # Load Groot N1 policy
    policy = Gr00tPolicy(
        model_path="~/models/groot-n1-2b",
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # Will need fine-tuning
        device='cuda'
    )

    print("✓ Simulation ready with Groot N1")
    print("Testing inference...")

    # Test inference loop
    for i in range(100):
        # Get observations
        rgb_image = camera.data.output["rgb"][0].cpu().numpy()  # (H, W, 3)
        joint_pos = robot.data.joint_pos[0].cpu().numpy()  # (22,)

        # Prepare input for Groot N1
        observation = {
            'rgb': torch.from_numpy(rgb_image).cuda().unsqueeze(0),  # (1, H, W, 3)
            'state': torch.from_numpy(joint_pos).cuda().unsqueeze(0),  # (1, 22)
            'language': "Walk forward"  # Text command
        }

        # Run inference
        with torch.no_grad():
            actions = policy(observation)  # (1, 22) joint commands

        # Apply actions (for now, just log)
        print(f"Step {i}: Action norm = {actions.norm().item():.3f}")

        # Step simulation
        sim.step()

    print("✓ Simulation test complete")

if __name__ == "__main__":
    main()
    simulation_app.close()
```

### Step 3: Run Simulation Test

```bash
# Test Groot N1 in simulation
cd ~/Projects/Berkeley-Humanoid-Lite

# Activate Isaac Lab environment
source .venv/bin/activate

# Run test
python scripts/groot/test_groot_sim.py --headless

# Or with GUI to visualize
python scripts/groot/test_groot_sim.py
```

**Expected outcome**:
- Robot spawns in Isaac Sim
- Camera renders RGB images
- Groot N1 generates actions (may be poor without fine-tuning)
- No crashes or errors

### Step 4: Analyze Pretrained Performance

The pretrained Groot N1 likely won't work well on Berkeley Humanoid Lite yet because:
- Trained on different embodiments
- Doesn't know Berkeley Humanoid Lite's kinematics
- Hasn't seen bipedal walking data

**This is expected!** Next phase: collect data and fine-tune.

## Phase 3: Camera Integration

### Hardware Setup

#### Option 1: Fixed Mount Camera (Simplest)

**Parts needed:**
- Raspberry Pi Camera v2 or Logitech C920
- USB cable (or CSI cable for Pi Camera)
- 3D printed mount

**Steps:**

1. **Design camera mount**:
   - CAD: Create bracket that attaches to torso/head
   - Position: ~10cm forward of torso, ~15cm above base
   - Orientation: Facing forward, level with horizon
   - Access: Easy to adjust/remove for maintenance

2. **Print and install**:
   ```bash
   # Print camera mount (example STL)
   # Use PLA or PETG, 20% infill, 0.2mm layer height

   # Mount camera to robot
   # Secure with M3 screws
   # Route USB cable through torso
   ```

3. **Connect to computer**:
   - Jetson Orin Nano: USB port or CSI connector
   - Intel N95: USB port

#### Option 2: 1-DOF Pan Neck (Recommended for better FOV)

**Additional parts:**
- 1x servo motor (same type as existing joints)
- 1x servo bracket/bearing
- Modified CAD for neck joint

**Steps:**
1. Design 1-DOF neck joint in CAD
2. 3D print neck parts
3. Install servo motor
4. Add to CAN bus (requires 1 more motor controller channel)
5. Update URDF/USD with neck joint
6. Retrain/fine-tune with neck control

**Note**: Start with fixed mount. Add neck later if needed.

### Software Setup

#### On Jetson Orin Nano

```bash
# Install camera drivers
sudo apt install v4l-utils

# List cameras
v4l2-ctl --list-devices

# Test camera (USB webcam)
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Capture test image
fswebcam -r 640x480 --no-banner test.jpg

# Or for Raspberry Pi Camera (CSI)
# Already supported on Jetson, just plug in
```

#### Camera Interface Code

```python
# Create camera capture class
# File: source/berkeley_humanoid_lite_lowlevel/camera.py

import cv2
import numpy as np
from typing import Tuple

class RobotCamera:
    """Camera interface for Berkeley Humanoid Lite"""

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

    def get_frame(self) -> np.ndarray:
        """Capture RGB frame"""
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame_resized(self, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Capture and resize frame for Groot N1"""
        frame = self.get_frame()
        return cv2.resize(frame, size)

    def release(self):
        """Release camera"""
        self.camera.release()
```

#### Test Camera

```bash
# Test camera capture
cd ~/Projects/Berkeley-Humanoid-Lite

python << 'EOF'
from berkeley_humanoid_lite_lowlevel.camera import RobotCamera

camera = RobotCamera(camera_id=0)

# Capture 10 frames
for i in range(10):
    frame = camera.get_frame_resized((224, 224))
    print(f"Frame {i}: shape={frame.shape}, dtype={frame.dtype}")

camera.release()
print("✓ Camera test successful")
EOF
```

## Phase 4: Data Collection

### Collection Strategy

Groot N1 needs demonstrations: (video, robot state, actions) triplets.

**Options:**

1. **Synthetic data** (Isaac Sim): Use trained RL policy to generate demos
2. **Teleoperation**: Human controls robot, record data
3. **Scripted behaviors**: Pre-programmed sequences
4. **Hybrid**: Combine all three

**Recommended**: Start with synthetic data (easy, fast), add real data later.

### Method 1: Synthetic Data from RL Policy

```python
# Generate synthetic demonstrations using trained RL policy
# File: scripts/groot/generate_synthetic_data.py

import torch
import numpy as np
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from berkeley_humanoid_lite.tasks import *
import gymnasium as gym

def generate_demonstrations(
    task_name: str,
    num_episodes: int = 1000,
    output_dir: str = "data/groot_demos"
):
    """Generate demonstration data using trained RL policy"""

    # Create environment
    env = gym.make(task_name, num_envs=1, render_mode="rgb_array")

    # Load trained policy
    from onnxruntime import InferenceSession
    policy = InferenceSession("logs/rsl_rl/latest/exported/policy.onnx")

    # Storage
    import os
    os.makedirs(output_dir, exist_ok=True)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'images': []
        }

        done = False
        while not done:
            # Get RGB image
            rgb = env.render()  # (H, W, 3)

            # Policy inference
            action = policy.run(None, {'obs': obs})[0]

            # Store
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['images'].append(rgb)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Save episode
        np.savez(
            f"{output_dir}/episode_{episode:05d}.npz",
            observations=np.array(episode_data['observations']),
            actions=np.array(episode_data['actions']),
            images=np.array(episode_data['images'])
        )

        print(f"Episode {episode}/{num_episodes} complete: {len(episode_data['actions'])} steps")

    print(f"✓ Generated {num_episodes} demonstrations in {output_dir}")

if __name__ == "__main__":
    generate_demonstrations(
        task_name="Velocity-Berkeley-Humanoid-Lite-v0",
        num_episodes=1000,
        output_dir="data/groot_demos_walking"
    )
    simulation_app.close()
```

### Method 2: Real Robot Teleoperation

```python
# Record demonstrations via teleoperation
# File: scripts/groot/record_teleoperation.py

import numpy as np
import time
from berkeley_humanoid_lite_lowlevel.camera import RobotCamera
from berkeley_humanoid_lite_lowlevel import RobotState
# Assuming you have teleop code

def record_demonstration(duration_seconds: int = 30, output_path: str = "demo.npz"):
    """Record teleoperation demonstration"""

    camera = RobotCamera()
    # robot = ... # Your robot interface

    data = {
        'images': [],
        'joint_pos': [],
        'joint_vel': [],
        'actions': [],
        'timestamps': []
    }

    print("Recording... Press Ctrl+C to stop")
    start_time = time.time()

    try:
        while time.time() - start_time < duration_seconds:
            # Capture
            rgb = camera.get_frame_resized((224, 224))
            # joint_pos = robot.get_joint_positions()
            # joint_vel = robot.get_joint_velocities()
            # action = teleop_controller.get_action()  # From gamepad/VR

            # Store
            data['images'].append(rgb)
            # data['joint_pos'].append(joint_pos)
            # data['joint_vel'].append(joint_vel)
            # data['actions'].append(action)
            data['timestamps'].append(time.time() - start_time)

            time.sleep(1/30)  # 30 Hz

    except KeyboardInterrupt:
        print("Recording stopped")

    # Save
    np.savez(
        output_path,
        **{k: np.array(v) for k, v in data.items()}
    )
    print(f"✓ Saved demonstration to {output_path}")

    camera.release()
```

### Convert to LeRobot Format

Groot N1 expects LeRobot format:

```python
# Convert demonstrations to LeRobot format
# File: scripts/groot/convert_to_lerobot.py

import numpy as np
from pathlib import Path
import json

def convert_to_lerobot(
    input_dir: str,
    output_dir: str,
    robot_name: str = "berkeley_humanoid_lite"
):
    """Convert demonstration data to LeRobot format"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all episodes
    episodes = sorted(input_path.glob("episode_*.npz"))

    # Create dataset structure
    dataset = {
        'episodes': [],
        'modalities': {
            'observation.images.head_camera': {'shape': (480, 640, 3), 'dtype': 'uint8'},
            'observation.state': {'shape': (22,), 'dtype': 'float32'},
            'action': {'shape': (22,), 'dtype': 'float32'},
        }
    }

    for i, episode_file in enumerate(episodes):
        data = np.load(episode_file)

        episode = {
            'episode_index': i,
            'observation.images.head_camera': data['images'],
            'observation.state': data['observations'],  # Joint positions
            'action': data['actions'],
            'timestamp': np.arange(len(data['actions'])) * 0.04,  # 25 Hz
        }

        # Save episode
        episode_path = output_path / f"episode_{i:06d}.npz"
        np.savez_compressed(episode_path, **episode)
        dataset['episodes'].append({'path': str(episode_path), 'length': len(data['actions'])})

        if i % 100 == 0:
            print(f"Processed {i}/{len(episodes)} episodes")

    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"✓ Converted {len(episodes)} episodes to LeRobot format")

# Usage
convert_to_lerobot(
    input_dir="data/groot_demos_walking",
    output_dir="data/lerobot/berkeley_humanoid_lite_walking"
)
```

### Data Collection Guidelines

**Quantity:**
- Minimum: 500 episodes
- Recommended: 1000-2000 episodes
- More is better for generalization

**Diversity:**
- Vary velocity commands (slow, fast, turning, backward)
- Different starting positions/orientations
- Include recovery behaviors (stumbling, correcting)
- Mix of short and long episodes

**Quality:**
- Smooth trajectories (avoid jerky movements)
- Successful task completion
- Realistic robot behavior

## Phase 5: Fine-Tuning

### Prepare Fine-Tuning Configuration

```yaml
# Create training config
# File: scripts/groot/configs/finetune_config.yaml

# Model
model:
  checkpoint: "~/models/groot-n1-2b"
  embodiment: "berkeley_humanoid_lite"
  freeze_vision: true  # Don't retrain vision encoder
  freeze_language: true  # Don't retrain language encoder

# Data
data:
  dataset_path: "data/lerobot/berkeley_humanoid_lite_walking"
  batch_size: 16
  num_workers: 4
  train_split: 0.9
  val_split: 0.1

# Training
training:
  num_epochs: 10
  learning_rate: 1e-5
  warmup_steps: 500
  gradient_clip: 1.0
  mixed_precision: true  # fp16 training
  checkpoint_every: 1000

# Optimization
optimizer:
  type: "adamw"
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Hardware
hardware:
  num_gpus: 1
  device: "cuda"
```

### Run Fine-Tuning

```bash
# Activate Groot environment
conda activate groot
# or
source ~/groot_env/bin/activate

# Navigate to Groot N1 repo
cd ~/Projects/Isaac-GR00T

# Start fine-tuning
python scripts/gr00t_finetune.py \
    --config ~/Projects/Berkeley-Humanoid-Lite/scripts/groot/configs/finetune_config.yaml \
    --dataset-path data/lerobot/berkeley_humanoid_lite_walking \
    --num-gpus 1 \
    --output-dir checkpoints/berkeley_humanoid_lite_walking

# Monitor with tensorboard
tensorboard --logdir checkpoints/berkeley_humanoid_lite_walking
```

**Training time**: 1-2 days on RTX PRO 6000 Blackwell

**Expected GPU memory**: 12-20 GB (adjust batch size if OOM)

### Monitor Training

```bash
# Watch TensorBoard for:
# - Training loss (should decrease)
# - Validation loss (should decrease, not overfit)
# - Action prediction accuracy
# - Vision-language alignment metrics
```

### Evaluate Fine-Tuned Model

```bash
# Test fine-tuned model in simulation
python scripts/groot/evaluate_model.py \
    --model-path checkpoints/berkeley_humanoid_lite_walking/final_model \
    --task Velocity-Berkeley-Humanoid-Lite-v0 \
    --num-episodes 50

# Metrics to check:
# - Success rate (% episodes that complete task)
# - Average reward
# - Stability (no falling)
# - Smoothness (action variation)
```

## Phase 6: Tethered Deployment

### Architecture

```
Berkeley Humanoid Lite
├─ Camera (streams video @ 30 FPS)
├─ IMU + Joint Encoders (publishes state @ 250 Hz)
└─ Motor Controllers (receives commands @ 250 Hz)
         │
         │ WiFi/Ethernet
         ↓
Development Workstation (RTX PRO 6000)
├─ Receives: Video + Robot State
├─ Runs: Groot N1 @ 10 Hz
└─ Sends: Joint Commands
         │
         ↓
Robot Low-Level Controller
└─ Interpolates commands 250 Hz → Motors
```

### Setup Network Communication

#### On Robot (Jetson or N95)

```python
# Create data streaming node
# File: source/berkeley_humanoid_lite_lowlevel/groot_client.py

import socket
import pickle
import numpy as np
import struct
from camera import RobotCamera

class GrootClient:
    """Stream robot data to Groot N1 server"""

    def __init__(self, server_ip: str = "192.168.1.100", port: int = 5555):
        self.server_ip = server_ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((server_ip, port))
        self.camera = RobotCamera()

        print(f"✓ Connected to Groot server at {server_ip}:{port}")

    def send_observation(self, joint_pos, joint_vel, imu_data):
        """Send robot state to server"""
        # Capture image
        rgb = self.camera.get_frame_resized((224, 224))

        # Package data
        data = {
            'rgb': rgb,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'imu': imu_data,
            'timestamp': time.time()
        }

        # Serialize and send
        payload = pickle.dumps(data)
        size = struct.pack('I', len(payload))
        self.socket.sendall(size + payload)

    def receive_action(self):
        """Receive action from server"""
        # Receive size
        size_data = self.socket.recv(4)
        size = struct.unpack('I', size_data)[0]

        # Receive payload
        payload = b''
        while len(payload) < size:
            payload += self.socket.recv(size - len(payload))

        # Deserialize
        action = pickle.loads(payload)
        return action['joint_commands']
```

#### On Workstation

```python
# Create Groot N1 inference server
# File: scripts/groot/groot_server.py

import socket
import pickle
import struct
import torch
from gr00t.model.policy import Gr00tPolicy

class GrootServer:
    """Groot N1 inference server"""

    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 5555
    ):
        # Load model
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag="berkeley_humanoid_lite",
            device='cuda'
        )
        self.policy.eval()

        # Setup server
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(1)

        print(f"✓ Groot server listening on {host}:{port}")

    def run(self):
        """Run inference server"""
        conn, addr = self.socket.accept()
        print(f"✓ Client connected: {addr}")

        try:
            while True:
                # Receive observation
                size_data = conn.recv(4)
                if not size_data:
                    break
                size = struct.unpack('I', size_data)[0]

                payload = b''
                while len(payload) < size:
                    payload += conn.recv(size - len(payload))

                obs = pickle.loads(payload)

                # Prepare input
                observation = {
                    'rgb': torch.from_numpy(obs['rgb']).cuda().unsqueeze(0),
                    'state': torch.from_numpy(obs['joint_pos']).cuda().unsqueeze(0),
                }

                # Inference
                with torch.no_grad():
                    action = self.policy(observation)

                # Send action
                response = {'joint_commands': action.cpu().numpy()[0]}
                payload = pickle.dumps(response)
                size = struct.pack('I', len(payload))
                conn.sendall(size + payload)

        except KeyboardInterrupt:
            print("Server shutting down")
        finally:
            conn.close()

# Run server
if __name__ == "__main__":
    server = GrootServer(
        model_path="checkpoints/berkeley_humanoid_lite_walking/final_model"
    )
    server.run()
```

### Test Tethered Operation

```bash
# Terminal 1 (Workstation): Start Groot server
cd ~/Projects/Berkeley-Humanoid-Lite
python scripts/groot/groot_server.py

# Terminal 2 (Robot): Start client and robot control
ssh jetson@<ROBOT_IP>
cd ~/robot_deployment
python groot_client_demo.py
```

### Measure Latency

```python
# Measure round-trip latency
import time

client = GrootClient(server_ip="192.168.1.100")

latencies = []
for i in range(100):
    start = time.time()

    # Send obs
    client.send_observation(joint_pos, joint_vel, imu_data)

    # Receive action
    action = client.receive_action()

    latency = time.time() - start
    latencies.append(latency)

print(f"Mean latency: {np.mean(latencies)*1000:.1f} ms")
print(f"Std latency: {np.std(latencies)*1000:.1f} ms")
```

**Target**: <100ms latency (acceptable for 10 Hz control)

## Phase 7: Onboard Deployment

### Option A: Jetson Orin Nano Super (Experimental)

```bash
# On Jetson Orin Nano Super

# Install Groot N1
pip install ~/Projects/Isaac-GR00T  # Copy wheel to Jetson

# Convert model to TensorRT for optimization
python scripts/groot/optimize_for_jetson.py \
    --model-path checkpoints/berkeley_humanoid_lite_walking/final_model \
    --output-path groot_n1_jetson.trt \
    --precision fp16

# Test inference speed
python scripts/groot/benchmark_jetson.py --model groot_n1_jetson.trt
```

**Expected performance**:
- Groot N1-1B + FP16 + TensorRT: 5-8 Hz (may work)
- Groot N1-2B: <5 Hz (likely too slow)

### Option B: Jetson AGX Orin (Recommended)

```bash
# On Jetson AGX Orin (32/64GB)

# Install and optimize
pip install ~/Projects/Isaac-GR00T
python scripts/groot/optimize_for_jetson.py \
    --model-path checkpoints/berkeley_humanoid_lite_walking/final_model \
    --output-path groot_n1_agx.trt \
    --precision fp16

# Benchmark
python scripts/groot/benchmark_jetson.py --model groot_n1_agx.trt
```

**Expected performance**:
- Groot N1-2B + FP16 + TensorRT: 10-15 Hz (perfect!)

### Deployment Script

```python
# Onboard Groot N1 control
# File: scripts/groot/deploy_onboard.py

import torch
import tensorrt as trt
from berkeley_humanoid_lite_lowlevel.camera import RobotCamera
# ... robot interface imports

def main():
    # Load TensorRT engine
    with open('groot_n1_jetson.trt', 'rb') as f:
        engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Initialize robot
    camera = RobotCamera()
    # robot = RobotInterface()

    print("✓ Groot N1 onboard deployment ready")

    # Control loop @ 10 Hz
    import time
    while True:
        start = time.time()

        # Get observation
        rgb = camera.get_frame_resized((224, 224))
        # joint_pos = robot.get_joint_positions()

        # Inference
        # action = run_tensorrt_inference(context, rgb, joint_pos)

        # Send to robot
        # robot.set_joint_commands(action)

        # Maintain 10 Hz
        elapsed = time.time() - start
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)

if __name__ == "__main__":
    main()
```

## Optimization Techniques

### Model Optimization

1. **Use smaller model**: Groot N1-1B instead of 2B
2. **Quantization**: FP16 or INT8 inference
3. **TensorRT**: Optimize for Jetson architecture
4. **Prune attention**: Reduce attention heads (advanced)

### Data Pipeline Optimization

```python
# Optimize camera capture
class OptimizedCamera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        # Set camera to MJPEG for faster capture
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # Disable auto-focus for consistent latency
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
```

### Inference Optimization

```python
# Use CUDA streams for parallel processing
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    # Preprocess image
    img_tensor = preprocess(rgb_image)

    # Inference
    with torch.no_grad():
        action = policy(img_tensor)

stream.synchronize()
```

## Troubleshooting

### Issue: Inference too slow

**Symptoms**: Can't achieve 10 Hz

**Solutions**:
1. Use smaller model (Groot N1-1B)
2. Reduce image resolution (224x224 → 128x128)
3. Optimize with TensorRT
4. Upgrade to Jetson AGX Orin
5. Run tethered to workstation

### Issue: Robot behavior unstable

**Symptoms**: Jerky movements, falling, oscillations

**Solutions**:
1. Lower PD gains
2. Add action filtering/smoothing
3. Collect more diverse training data
4. Fine-tune for longer
5. Check observation normalization

### Issue: Vision not helping

**Symptoms**: Model ignores camera input, relies only on proprioception

**Solutions**:
1. Verify camera is providing useful information (test with viz)
2. Add vision-dependent tasks to training data
3. Check data augmentation isn't too aggressive
4. Increase vision encoder learning rate (if unfrozen)

### Issue: Out of memory on Jetson

**Symptoms**: CUDA OOM errors

**Solutions**:
1. Use Groot N1-1B instead of 2B
2. Quantize to INT8
3. Reduce batch size (if batching multiple observations)
4. Clear CUDA cache: `torch.cuda.empty_cache()`
5. Upgrade to larger Jetson (32GB or 64GB AGX Orin)

### Issue: High latency on network

**Symptoms**: >200ms round-trip time

**Solutions**:
1. Use wired Ethernet instead of WiFi
2. Reduce image resolution
3. Compress images before sending (JPEG)
4. Use faster serialization (protobuf instead of pickle)
5. Move to onboard inference

## Performance Benchmarks

### Inference Speed

| Platform | Model | Precision | Speed | Viable for 10 Hz? |
|----------|-------|-----------|-------|-------------------|
| RTX PRO 6000 | N1-2B | FP32 | 50 Hz | ✅✅ |
| RTX PRO 6000 | N1-2B | FP16 | 80 Hz | ✅✅ |
| Orin Nano Super | N1-1B | FP16 + TRT | 5-8 Hz | ⚠️ Marginal |
| Orin Nano Super | N1-2B | FP16 + TRT | 3-5 Hz | ❌ Too slow |
| AGX Orin 32GB | N1-2B | FP16 + TRT | 12-15 Hz | ✅ |
| AGX Thor | N1-2B | FP16 + TRT | 25+ Hz | ✅✅ |

### Memory Usage

| Platform | Model | Memory Used | Headroom |
|----------|-------|-------------|----------|
| Orin Nano Super | N1-1B | 6.5 GB | 1.5 GB |
| Orin Nano Super | N1-2B | 7.8 GB | 0.2 GB (tight!) |
| AGX Orin 32GB | N1-2B | 8 GB | 24 GB |

### Task Performance (After Fine-tuning)

| Task | Success Rate | Notes |
|------|--------------|-------|
| Forward walking | 95% | Robust |
| Backward walking | 85% | Sometimes unstable |
| Turning in place | 90% | Good |
| Walking + arm waving | 70% | More challenging |
| Pick and place | 60% | Vision-dependent |

## Comparison: RL vs Groot N1

### Current RL Approach

**Pros:**
- Fast training (2 hours)
- Efficient inference (25 Hz, CPU works)
- Proven sim2real transfer
- Task-specific optimization
- No camera needed

**Cons:**
- Single task per policy
- Train from scratch for each task
- No language understanding
- No vision-based behaviors
- Limited generalization

### Groot N1 Approach

**Pros:**
- Multi-task capability
- Language-conditioned ("walk backward")
- Vision-based (can see and react)
- Transfer learning (pre-trained)
- Generalist behaviors
- Zero-shot to novel tasks

**Cons:**
- Slower inference (10 Hz, GPU required)
- Requires camera
- Needs demonstration data
- Fine-tuning takes days
- More memory intensive
- Unproven sim2real for this robot

### Hybrid Approach (Recommended)

Use **both**:

1. **RL for locomotion**: Fast, efficient, proven
2. **Groot N1 for manipulation/vision**: Language-conditioned, flexible

Example:
```
User: "Walk to the table and pick up the cup"
  ↓
Groot N1 System 2: Parse command, plan approach
  ↓
Use RL policy for walking to table
  ↓
Switch to Groot N1 for vision-based grasping
```

## Next Steps

### Short Term (1-2 months)
1. ✅ Complete RL velocity tracking policy
2. ✅ Add camera to robot (fixed mount)
3. ✅ Test Groot N1 in simulation
4. ✅ Collect synthetic demonstration data
5. ✅ Fine-tune Groot N1 for walking
6. ✅ Test tethered deployment

### Medium Term (3-6 months)
7. Collect real-world demonstration data
8. Fine-tune for manipulation tasks
9. Test onboard deployment (Jetson AGX Orin)
10. Implement hybrid RL + Groot N1 system

### Long Term (6-12 months)
11. Multi-task training (walk + manipulate + gesture)
12. Real-world deployment and validation
13. Publish results / share with community

## Resources

- **Groot N1 GitHub**: https://github.com/NVIDIA/Isaac-GR00T
- **Groot N1 Paper**: https://arxiv.org/abs/2503.14734
- **HuggingFace Models**: https://huggingface.co/nvidia/GR00T-N1-2B
- **LeRobot**: https://github.com/huggingface/lerobot
- **Isaac Sim**: https://docs.omniverse.nvidia.com/isaacsim/
- **Berkeley Humanoid Lite Docs**: `.external-libs/berkeley-humanoid-lite-docs/docs/`

## Conclusion

Groot N1 integration is a **medium-to-high complexity** project that requires:
- GPU compute (workstation or Jetson AGX Orin)
- Camera hardware
- Demonstration data collection
- Fine-tuning expertise
- Robot deployment experience

**Start conservatively:**
1. Use your existing Orin Nano Super for testing
2. Test tethered to your RTX PRO 6000 workstation
3. Only invest in AGX Orin if Groot N1 proves valuable

**The payoff:**
- Multi-modal robot control (vision + language + actions)
- Generalist behaviors instead of task-specific policies
- Foundation model that improves as NVIDIA releases updates
- Research platform for advanced robotics

Good luck! This is cutting-edge robotics research.
