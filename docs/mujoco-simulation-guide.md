# MuJoCo Simulation Guide

This guide covers using MuJoCo for sim-to-sim validation of trained policies on the Berkeley Humanoid Lite robot.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Gamepad Controller Support](#gamepad-controller-support)
4. [Debug Mode](#debug-mode)
5. [Troubleshooting](#troubleshooting)
6. [Technical Details](#technical-details)

## Overview

MuJoCo provides a lightweight physics simulation environment for validating trained policies before deploying to the real robot. Unlike Isaac Sim (used for training), MuJoCo:

- Runs on modest hardware without GPU requirements
- Provides real-time visualization
- Serves as an intermediate validation step (sim-to-sim)
- Supports gamepad control for teleoperation

**Simulation Pipeline:**
```
Isaac Sim Training → Policy Export → MuJoCo Validation → Real Robot Deployment
```

## Quick Start

### Prerequisites

1. Trained and exported policy (see [Training Guide](training-guide.md))
2. Policy configuration YAML in `configs/` directory
3. Connected gamepad (PS5 DualSense, Xbox, or XInput-compatible)

### Running the Simulation

**Basic usage:**
```bash
python scripts/sim2sim/play_mujoco.py --config configs/policy_latest.yaml
```

**With debug logging:**
```bash
python scripts/sim2sim/play_mujoco.py --config configs/policy_latest.yaml --debug
```

### Control Modes

The robot has three control modes that can be switched via gamepad:

| Mode | Description | Button Combination |
|------|-------------|-------------------|
| 1 - Idle | Robot stands still, motors relaxed | **X** or **L3/R3 press** |
| 2 - Init | Robot moves to default standing pose | **A + L1** (Left Bumper) |
| 3 - RL Control | Neural network policy controls locomotion | **A + R1** (Right Bumper) |

**Default mode:** RL Control (mode 3)

### Locomotion Controls

When in RL Control mode (mode 3):

| Action | Control |
|--------|---------|
| Walk forward/backward | **Left stick** up/down |
| Turn left/right | **Left stick** left/right |
| Strafe left/right | **Right stick** left/right |

## Gamepad Controller Support

### Supported Controllers

The system automatically detects and configures the following controller types:

1. **Sony DualSense (PS5)** - Recommended
2. **Xbox/XInput controllers** - Standard Xbox controllers
3. **Generic gamepads** - Falls back to XInput profile

### Auto-Detection

The gamepad module automatically:
- Detects connected controller by device name
- Selects the appropriate normalization profile
- Handles different axis value ranges (8-bit vs 16-bit)
- Filters out non-gamepad devices (motion sensors, touchpads, keyboards with gamepad interfaces)

Example output:
```
Detected gamepad: Sony Interactive Entertainment DualSense Wireless Controller
Using controller: Sony DualSense (PS5)
```

### Technical Implementation

Different controllers report analog stick values in different ranges:

| Controller Type | Center Value | Range | Normalization |
|----------------|--------------|-------|---------------|
| DualSense (PS5) | 128 | 0-255 (8-bit) | `(value - 128) / 128` |
| XInput/Xbox | 0 | -32768 to 32767 (16-bit) | `value / 32768` |

The system uses a **Controller Profile** architecture that encapsulates:
- Center value (neutral stick position)
- Maximum range for normalization
- Axis inversion settings
- Dead zone filtering (default: 1% to prevent drift)

### Adding New Controller Types

To add support for a new controller type:

1. Add detection logic in `gamepad.py`:
```python
# In _detect_controller() method
if 'new_controller_name' in device_name:
    return device, CONTROLLER_PROFILES['new_controller']
```

2. Define controller profile:
```python
CONTROLLER_PROFILES['new_controller'] = ControllerProfile(
    name='New Controller Name',
    center_value=128.0,    # Adjust based on controller
    max_range=128.0,       # Adjust based on controller
    invert=True
)
```

## Debug Mode

Debug mode enables detailed logging of gamepad inputs and simulation state.

### Enabling Debug Mode

```bash
python scripts/sim2sim/play_mujoco.py --config configs/policy_latest.yaml --debug
```

### Debug Output

When enabled, you'll see:

**Gamepad controller detection:**
```
Detected gamepad: Sony Interactive Entertainment DualSense Wireless Controller
Using controller: Sony DualSense (PS5)
```

**Simulation state (every 50 steps):**
```
Step 0: mode=3, vx=0.00, vy=0.00, vyaw=0.00
Step 50: mode=3, vx=0.45, vy=-0.12, vyaw=0.23
Step 100: mode=3, vx=0.67, vy=0.00, vyaw=-0.15
```

**Gamepad errors (if any):**
```
Gamepad error: [Errno 19] No such device
```

### Use Cases for Debug Mode

- **Controller troubleshooting:** Verify gamepad is detected and sending values
- **Policy validation:** Monitor command velocities sent to the policy
- **Mode switching issues:** Confirm mode changes respond to button presses
- **Performance analysis:** Check simulation frequency and timing

## Troubleshooting

### Issue: Robot Not Moving

**Symptoms:**
- Robot stands still despite moving gamepad sticks
- Velocity values remain at 0.00

**Possible Causes:**

1. **Wrong control mode**
   - Solution: Press **A + R1** to enter RL Control mode (mode 3)

2. **Gamepad not detected**
   - Run with `--debug` flag to check detection
   - Verify gamepad is connected: `ls /dev/input/js*`
   - Try reconnecting the controller

3. **Multiple input devices conflict**
   - Some keyboards (e.g., Keychron) present as gamepads
   - The system prioritizes DualSense/Xbox controllers
   - Unplug conflicting devices if detection fails

4. **Dead zone too large**
   - Default dead zone is 1% (0.01)
   - Small stick movements may be filtered out
   - Adjust in gamepad.py: `Se2Gamepad(dead_zone=0.005)`

### Issue: Mesh Loading Errors

**Symptoms:**
```
ValueError: Error opening file 'source/.../assets/source/.../meshes/...'
```

**Cause:** Incorrect mesh path resolution in MJCF files (fixed in recent update)

**Solution:** Ensure you have the latest MJCF files with corrected mesh paths:
- `meshdir="."` in `<compiler>` directive
- Mesh paths: `../meshes/*.stl` (relative to MJCF file location)

See [Technical Details - MJCF Mesh Path Fix](#mjcf-mesh-path-fix) below.

### Issue: Wrong Controller Detected

**Symptoms:**
```
Using controller: XInput/Xbox Controller
```
But you're using a PS5 controller.

**Cause:** Controller not recognized by name detection logic

**Solution:**
1. Check device name: `python -c "from inputs import devices; [print(d.name) for d in devices]"`
2. Add detection pattern to `gamepad.py`:
```python
if 'your_controller_keyword' in device_name:
    return device, CONTROLLER_PROFILES['dualsense']
```

### Issue: Simulation Too Slow/Fast

**Check frequencies:**
```python
Policy frequency:  25.0 Hz
Physics frequency: 2000.0 Hz
Physics substeps:  80
```

**These are controlled by `configs/policy_latest.yaml`:**
- `policy_dt: 0.04` → 25 Hz policy inference
- `physics_dt: 0.0005` → 2000 Hz physics simulation
- Substeps = policy_dt / physics_dt

**For real-time operation:** Ensure your CPU can maintain these frequencies. Lower values if simulation stutters.

## Technical Details

### MJCF Mesh Path Fix

**Problem:** Original MJCF files had incorrect mesh path configuration that caused path resolution errors when loading robot models in MuJoCo.

**Root Cause:**
- `meshdir="assets"` in compiler directive
- Mesh paths: `merged/*.stl`
- MuJoCo concatenated paths: `mjcf_dir/assets/merged/*.stl` (incorrect)

**Solution Applied:**
Changed both `berkeley_humanoid_lite.xml` and `berkeley_humanoid_lite_biped.xml`:

```xml
<!-- BEFORE -->
<compiler angle="radian" meshdir="assets" autolimits="true"/>
<asset>
  <mesh file="merged/leg_right_knee_pitch_visual.stl"/>
  ...
</asset>

<!-- AFTER -->
<compiler angle="radian" meshdir="." autolimits="true"/>
<asset>
  <mesh file="../meshes/leg_right_knee_pitch_visual.stl"/>
  ...
</asset>
```

**Path Resolution:**
- MJCF file location: `data/robots/berkeley_humanoid/berkeley_humanoid_lite/mjcf/`
- Mesh directory: `data/robots/berkeley_humanoid/berkeley_humanoid_lite/meshes/`
- Relative path: `../meshes/` (up one directory, then into meshes/)

**Verification:**
```bash
# Test that MuJoCo can load the model
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite/mjcf/bhl_scene.xml'); print('Success')"
```

### Multi-Rate Control Loop

The simulation uses different update frequencies for different components:

```python
physics_dt:  0.0005  # 2000 Hz - Physics simulation timestep
control_dt:  0.004   # 250 Hz  - Hardware motor control (real robot only)
policy_dt:   0.04    # 25 Hz   - Neural network policy inference
```

**Policy Decimation:**
```python
physics_substeps = policy_dt / physics_dt  # 80 substeps
for _ in range(physics_substeps):
    apply_PD_control(actions)
    mujoco.mj_step()  # Advance physics by 0.0005s
```

This matches the real robot's control architecture where fast PD control runs at 250 Hz while the policy updates at 25 Hz.

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│           MujocoSimulator                       │
│  ┌──────────────────────────────────────────┐   │
│  │  Se2Gamepad (threaded)                   │   │
│  │  - Auto-detect controller                │   │
│  │  - Read gamepad events                   │   │
│  │  - Normalize axis values                 │   │
│  │  - Update command buffer                 │   │
│  └──────────────────────────────────────────┘   │
│                    │                             │
│                    ▼                             │
│  ┌──────────────────────────────────────────┐   │
│  │  RlController                            │   │
│  │  - Load ONNX policy                      │   │
│  │  - Inference (25 Hz)                     │   │
│  └──────────────────────────────────────────┘   │
│                    │                             │
│                    ▼                             │
│  ┌──────────────────────────────────────────┐   │
│  │  PD Control + MuJoCo Physics (2000 Hz)   │   │
│  │  - Compute joint torques                 │   │
│  │  - Step physics simulation               │   │
│  └──────────────────────────────────────────┘   │
│                    │                             │
│                    ▼                             │
│  ┌──────────────────────────────────────────┐   │
│  │  MuJoCo Viewer                           │   │
│  │  - Real-time 3D visualization            │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Configuration Files

### Policy Configuration (policy_latest.yaml)

Generated automatically by `play.py` after training. Contains:

```yaml
policy_checkpoint_path: path/to/policy.onnx
num_joints: 22  # or 12 for biped
joints: [list of joint names in order]
joint_kp: [PD proportional gains]
joint_kd: [PD derivative gains]
effort_limits: [torque limits per joint]
action_indices: [which joints to actuate]
default_joint_positions: [standing pose]
# ... more parameters
```

**Important:** This file must match the trained policy's configuration exactly.

### Robot Variants

| Variant | DOF | Config | MJCF Scene |
|---------|-----|--------|------------|
| Humanoid | 22 | num_joints: 22 | bhl_scene.xml |
| Biped | 12 | num_joints: 12 | bhl_biped_scene.xml |

The system automatically selects the correct MJCF model based on `num_joints` in the config.

## Related Documentation

- [Training Guide](training-guide.md) - Train policies in Isaac Sim
- [Policies Guide](policies-guide.md) - Export and manage policies
- [CLAUDE.md](../CLAUDE.md) - Complete codebase documentation

## Changelog

### 2025-10-23
- Added multi-controller support with auto-detection
- Fixed PS5 DualSense normalization (8-bit vs 16-bit)
- Fixed MJCF mesh path resolution
- Added debug mode for troubleshooting
- Added command-line arguments to play_mujoco.py

### Earlier
- Initial MuJoCo integration
- Basic gamepad support
- Real-time visualization
