# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Berkeley Humanoid Lite is an open-source, sub-$5,000 humanoid robot platform built on Isaac Lab. This workspace uses a **modular architecture with three independent submodules** that work together:

1. **`source/berkeley_humanoid_lite/`** - Isaac Lab training environments and task definitions
2. **`source/berkeley_humanoid_lite_assets/`** - Robot descriptions (URDF/MJCF/USD) and generation tools
3. **`source/berkeley_humanoid_lite_lowlevel/`** - Real robot deployment code (C control loop + Python policy inference)

All commands should be run from the **root directory** unless otherwise specified. Entry points are in `scripts/`.

## Development Commands

### Environment Setup

```bash
# Install all dependencies (uses uv workspace)
uv sync

# Initialize git submodules (if not already done)
git submodule update --init --recursive

# Sync documentation from GitBook
uv run --directory .external-libs/berkeley-humanoid-lite-docs sync-docs
```

### Training Workflow

```bash
# Train a policy (uses Isaac Sim)
python scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --num_envs 2048 \
    --video  # optional: record training videos

# Train humanoid (full body, 22 DOF)
python scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-Humanoid-v0 \
    --num_envs 2048

# Evaluate and export policy (generates ONNX + deployment config)
python scripts/rsl_rl/play.py \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --video
```

**Important**: `play.py` automatically exports:
- Policy checkpoints in ONNX and TorchScript formats
- Deployment config to `configs/policy_latest.yaml` (contains joint order, gains, effort limits)

### Simulation & Validation

```bash
# Test policy in MuJoCo (lightweight sim-to-sim validation)
python scripts/sim2sim/play_mujoco.py --config configs/policy_biped_25hz_a.yaml

# Visualize real robot state (receives UDP from robot)
python scripts/sim2real/visualize.py

# Manual teleoperation (gamepad control via IK)
python scripts/teleop/run_teleop.py
```

### Linting & Code Quality

```bash
# Run flake8 (max line length 120, Google-style docstrings)
flake8 .

# Organize imports (Black-compatible, custom Isaac Lab section)
isort .

# Type checking (basic mode, Python 3.11)
pyright
```

### Docker

```bash
# Build and run Isaac Sim environment
cd docker
docker-compose up -d

# Access container
docker exec -it <container-name> bash
```

## Architecture & Key Patterns

### Workspace Structure

This is a **uv workspace** with unified dependency management across all submodules. The root `pyproject.toml` defines workspace members and shared dependencies (PyTorch, Isaac Sim, Isaac Lab, etc.).

**Critical Dependencies**:
- Python 3.11 (required for Isaac Sim 5.0 / Isaac Lab 2.2)
- PyTorch 2.7.0 with CUDA 12.8 (supports Blackwell GPU architecture)
- Isaac Sim 5.0.0 & Isaac Lab 2.2.0
- MuJoCo (for deployment simulation)
- ONNX Runtime GPU 1.22.0+ (for GPU-accelerated policy inference)

### Manager-Based RL Environment Pattern

All tasks follow the Isaac Lab manager-based structure:

```python
@configclass
class MyEnvCfg(LocomotionVelocityEnvCfg):
    # Define modular MDP components
    observations: ObservationsCfg  # State processing
    actions: ActionsCfg            # Joint position targets
    rewards: RewardsCfg            # Task objectives
    terminations: TerminationsCfg  # Episode end conditions
    events: EventCfg               # Randomization/resets
    curriculums: CurriculumCfg     # Training curriculum
```

Environment registration uses Gymnasium:
```python
gym.register(
    id="Velocity-Berkeley-Humanoid-Lite-Biped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": BerkeleyHumanoidLiteBipedEnvCfg, ...}
)
```

### Configuration-Driven Deployment

Training automatically generates deployment configs to ensure consistency:

**During `play.py`**:
```python
# Extracts from training environment
deploy_config = {
    "policy_checkpoint_path": "policy.onnx",
    "num_joints": num_joints,
    "joints": joint_names,           # Joint order
    "joint_kp": joint_kp.tolist(),   # From actuator groups
    "joint_kd": joint_kd.tolist(),
    "effort_limits": effort_limits.tolist(),
    "action_indices": action_indices,  # Which joints to actuate
    # ... more
}
OmegaConf.save(deploy_config, "configs/policy_latest.yaml")
```

**On robot** (lowlevel):
```python
cfg = Cfg.from_arguments()  # Loads YAML
controller = RlController(cfg)  # ONNX inference
```

### Dual Simulator Architecture

**Isaac Sim** (training):
- GPU-accelerated parallel environments (2048+ envs)
- Photorealistic rendering for sim-to-real transfer
- USD format assets

**MuJoCo** (deployment/validation):
- Lightweight, runs on modest hardware
- Real-time policy testing before robot deployment
- MJCF format assets

All robot descriptions maintained in 3 formats: URDF → MJCF/USD via conversion scripts in `berkeley_humanoid_lite_assets`.

### Multi-Rate Control Loop

The system uses different update frequencies:

```yaml
physics_dt: 0.0005    # 2000 Hz - Physics simulation step
control_dt: 0.004     # 250 Hz - Hardware motor control (real robot)
policy_dt: 0.04       # 25 Hz - Neural network policy inference (10x slower)
```

Policy runs at lower frequency with decimation:
```python
physics_substeps = int(policy_dt / physics_dt)  # e.g., 80 steps
for _ in range(physics_substeps):
    apply_actions()
    mujoco.mj_step()
```

### Action Indices Concept

Allows flexible joint selection per robot variant:

```python
# Biped: only actuate 12 leg joints (skip 10 arm joints)
action_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Maps 12-dim action vector to correct joints:
target_positions[self.cfg.action_indices] = actions
```

### Submodule Integration

**Assets → Training**:
```python
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_BIPED_CFG
scene.robot = HUMANOID_LITE_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
```

**Training → Lowlevel**:
- Training exports ONNX policy + deployment YAML
- Lowlevel loads ONNX for inference on real robot

**Submodules are separate git repositories**:
- Assets: `https://github.com/kranthie/Berkeley-Humanoid-Lite-Assets.git`
- Lowlevel: `https://github.com/kranthie/Berkeley-Humanoid-Lite-Lowlevel.git`
- Each has `origin` (fork) and `upstream` (HybridRobotics) remotes

## Robot Configurations

### Two Main Variants

1. **Biped** (12 DOF legs only):
   - Task: `Velocity-Berkeley-Humanoid-Lite-Biped-v0`
   - Config: `source/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/`
   - Asset: `HUMANOID_LITE_BIPED_CFG`

2. **Humanoid** (22 DOF full body):
   - Task: `Velocity-Berkeley-Humanoid-Lite-Humanoid-v0`
   - Config: `source/berkeley_humanoid_lite/tasks/locomotion/velocity/config/humanoid/`
   - Asset: `HUMANOID_LITE_CFG`

### Joint Naming Convention

Pattern: `{side}_{part}_{axis}_{type}_joint`

Examples:
- `leg_left_hip_pitch_joint`
- `arm_right_shoulder_roll_joint`
- `leg_right_ankle_yaw_joint`

Axes: `pitch`, `roll`, `yaw`

## Common Development Workflow

### 1. Modify Task Configuration

Edit reward functions, observations, or other MDP components:
```
source/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/
├── __init__.py
├── agents/              # PPO hyperparameters
├── env_cfg.py          # Environment configuration
└── mdp_cfg.py          # Rewards, terminations, observations
```

### 2. Train Policy

```bash
python scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --num_envs 2048
```

Outputs:
- Logs: `logs/rsl_rl/{experiment_name}/{timestamp}/`
- Checkpoints: `logs/rsl_rl/{experiment_name}/{timestamp}/checkpoints/`
- TensorBoard logs for monitoring

### 3. Evaluate and Export

```bash
python scripts/rsl_rl/play.py \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0
```

Generates:
- `{log_dir}/exported/policy.onnx` - Deployable policy
- `{log_dir}/exported/policy.pt` - TorchScript version
- `configs/policy_latest.yaml` - Deployment configuration

### 4. Validate in MuJoCo

```bash
python scripts/sim2sim/play_mujoco.py --config configs/policy_latest.yaml
```

Opens real-time MuJoCo visualization with policy running.

### 5. Deploy to Robot (if applicable)

```bash
# Copy to lowlevel submodule
cp configs/policy_latest.yaml source/berkeley_humanoid_lite_lowlevel/configs/
cp {log_dir}/exported/policy.onnx source/berkeley_humanoid_lite_lowlevel/checkpoints/

# On robot (C control loop + Python policy)
cd source/berkeley_humanoid_lite_lowlevel
make run
```

## Code Style Conventions

### Import Organization (isort)

```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import os, sys

# 3. Third-party
import numpy as np
import torch

# 4. IsaacLab party (custom section)
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

# 5. First-party (this project)
from berkeley_humanoid_lite_assets import HUMANOID_LITE_BIPED_CFG

# 6. Local
from . import env_cfg
```

### Type Hints

Use throughout, with `TYPE_CHECKING` blocks for circular imports:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from berkeley_humanoid_lite.tasks import LocomotionVelocityEnvCfg
```

### Configuration Classes

Use `@configclass` decorator for all configs:
```python
from isaaclab.utils import configclass

@configclass
class MyRewardsCfg:
    track_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        params={"std": 0.25},
        weight=2.0
    )
```

## Asset Generation (Advanced)

Robot descriptions are generated from Onshape CAD:

```bash
cd source/berkeley_humanoid_lite_assets

# Export URDF from Onshape
python scripts/export_onshape_to_urdf.py

# Convert URDF to MJCF
python scripts/export_onshape_to_mjcf.py

# Convert URDF to USD (for Isaac Sim)
python scripts/convert_urdf_to_usd.py
```

Generated files stored in:
```
data/robots/berkeley_humanoid/berkeley_humanoid_lite/
├── urdf/
├── mjcf/
├── usd/
└── meshes/
```

## Troubleshooting

### GPU/CUDA Issues
- Ensure CUDA 12.8 is installed (required by PyTorch 2.7.0)
- Isaac Sim requires GPU with compute capability ≥ 7.0
- Blackwell GPUs (sm_120) now supported with PyTorch 2.7.0+

### Import Errors
- Run `uv sync` to ensure all workspace members are installed
- Check that you're in the root directory when running scripts

### Submodule Issues
```bash
# Re-sync submodules if they're out of date
git submodule sync
git submodule update --init --recursive
```

### Policy Export Fails
- Always run `play.py` after training to generate deployment configs
- Check that checkpoint path exists in `logs/rsl_rl/{experiment}/`

## Documentation

- **Official Docs**: https://berkeley-humanoid-lite.gitbook.io/docs
- **Paper**: https://arxiv.org/abs/2504.17249
- **Local Docs**: Run `uv run --directory .external-libs/berkeley-humanoid-lite-docs sync-docs` to download latest docs to `.external-libs/berkeley-humanoid-lite-docs/docs/`

## Git Workflow

This is a **fork** of the original HybridRobotics repository:

- **origin**: `https://github.com/kranthie/Berkeley-Humanoid-Lite.git` (your fork)
- **upstream**: `https://github.com/HybridRobotics/Berkeley-Humanoid-Lite` (original)

Submodules also follow this pattern (each with origin/upstream remotes).

To sync with upstream:
```bash
git fetch upstream
git merge upstream/main
git submodule update --init --recursive
```
