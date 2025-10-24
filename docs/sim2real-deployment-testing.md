# Sim2Real Deployment Testing Guide

This guide provides a comprehensive checklist for deploying trained policies to the real Berkeley Humanoid Lite robot and verifying functionality.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Steps](#deployment-steps)
3. [Testing Procedures](#testing-procedures)
4. [Open Items & Verification Needed](#open-items--verification-needed)
5. [Troubleshooting](#troubleshooting)
6. [Safety Protocols](#safety-protocols)

## Pre-Deployment Checklist

### Software Preparation

- [ ] Policy trained successfully in Isaac Sim
- [ ] Policy exported to ONNX format via `play.py`
- [ ] Deployment config generated: `configs/policy_latest.yaml`
- [ ] Policy validated in MuJoCo sim2sim environment
- [ ] Robot moves correctly with gamepad in MuJoCo
- [ ] All submodules synced to latest versions

### Hardware Preparation

- [ ] Robot fully assembled and secured
- [ ] All 22 motors (or 12 for biped) connected via CAN bus
- [ ] IMU connected and tested
- [ ] On-board computer (NUC) powered and accessible
- [ ] USB-CAN adapters connected (can0, can1, can2, can3)
- [ ] Gamepad controller connected to NUC
- [ ] Safety tether/kill switch accessible
- [ ] Testing area cleared and safe

### On-board Computer Setup

- [ ] Ubuntu 22.04 installed
- [ ] Dependencies installed: `build-essential`, `cmake`, `net-tools`, `can-utils`, `python3-pip`, `libboost-all-dev`
- [ ] uv package manager installed
- [ ] Repository cloned and submodules initialized
- [ ] C codebase built successfully
- [ ] Python environment set up with `uv sync`
- [ ] CAN transports script tested: `source ./scripts/start_can_transports.sh`

## On-board Computer Initial Setup

This section covers the one-time setup of the on-board computer. Skip to [Deployment Steps](#deployment-steps) if your on-board computer is already configured.

### 1. Install Ubuntu 22.04

Follow the [Ubuntu installation tutorial](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview) to install Ubuntu 22.04 on the on-board computer (NUC or similar).

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake net-tools can-utils python3-pip libboost-all-dev
```

**Key dependencies:**
- `build-essential` - C/C++ compiler toolchain
- `cmake` - Build system for C codebase
- `net-tools` - Network utilities
- `can-utils` - CAN bus utilities for motor communication
- `python3-pip` - Python package installer
- `libboost-all-dev` - Boost C++ libraries (required for C codebase)

### 3. Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:
```bash
source $HOME/.local/bin/env
```

Verify installation:
```bash
uv --version
```

### 4. Clone Repository and Initialize Submodules

```bash
cd ~
mkdir -p Projects
cd Projects
git clone https://github.com/YOUR_USERNAME/Berkeley-Humanoid-Lite.git
cd Berkeley-Humanoid-Lite
git submodule update --init --recursive
```

**Important:** Replace `YOUR_USERNAME` with your GitHub username if using a fork.

### 5. Build C Codebase

The C codebase provides real-time control with better performance for locomotion tasks.

```bash
cd source/berkeley_humanoid_lite_lowlevel
make clean
make
```

**Expected output:** Executable created at `build/main`

Verify the build:
```bash
ls -lh build/main
file build/main
```

### 6. Set Up Python Environment

Set up Python dependencies for the lowlevel module:

```bash
cd source/berkeley_humanoid_lite_lowlevel
uv sync
```

This will:
- Download Python 3.11 (if not already installed)
- Create a virtual environment at `../../.venv`
- Install all required dependencies (onnxruntime, python-can, inputs, etc.)

**Expected output:** All packages installed successfully

Verify the setup:
```bash
uv run python -c "from berkeley_humanoid_lite_lowlevel.policy.gamepad import Se2Gamepad; print('Setup successful')"
```

### 7. Test CAN Transports (Hardware Required)

**Note:** This step requires the robot hardware to be connected.

```bash
cd ~/Projects/Berkeley-Humanoid-Lite
source ./scripts/start_can_transports.sh
```

Verify CAN interfaces are up:
```bash
ip link show can0
ip link show can1
ip link show can2
ip link show can3
```

To stop CAN transports:
```bash
source ./scripts/stop_can_transports.sh
```

### 8. Setup Verification Summary

Confirm all components are ready:

- [ ] C binary exists: `~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/build/main`
- [ ] Python modules import correctly
- [ ] CAN interfaces activate without errors
- [ ] Virtual environment created at `~/Projects/Berkeley-Humanoid-Lite/.venv`

## Deployment Steps

### Step 1: Copy Policy Files to Robot

Transfer the trained policy and configuration to the on-board computer's lowlevel submodule:

```bash
# On development machine, from repository root
scp configs/policy_latest.yaml robot@nuc:~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/configs/

# Find your latest exported policy and copy it
scp logs/rsl_rl/humanoid/YYYY-MM-DD_HH-MM-SS/exported/policy.onnx robot@nuc:~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/checkpoints/
```

**Notes:**
- Replace `robot@nuc` with your actual username and hostname
- Replace `YYYY-MM-DD_HH-MM-SS` with your training run timestamp
- Policy files must go to the **lowlevel submodule**, not the main repository directory
- Both C and Python codebases load policies from the lowlevel submodule

Verify files were copied:
```bash
# On robot
ls -lh ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/configs/policy_latest.yaml
ls -lh ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/checkpoints/policy.onnx
```

### Step 2: Verify Motor Connections

On the robot's on-board computer:

```bash
# Start CAN transports
source ./scripts/start_can_transports.sh

# Test individual motor
python ./source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/motor/ping.py --port can0 --id 1

# Test all motors
python ./source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/robot/anyonehere.py
```

Expected output: All motors should respond with their status.

### Step 3: Verify IMU Connection

```bash
python ./source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/robot/test_imu.py
```

Expected output: IMU orientation and angular velocity readings.

### Step 4: Verify Gamepad Connection

```bash
python ./source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/policy/udp_joystick.py
```

Expected output: Joystick readings when moving sticks and pressing buttons.

**⚠️ IMPORTANT:** See [Open Items - Gamepad Controller Verification](#gamepad-controller-verification) below.

### Step 5: Run Policy on Robot

**Before running**, ensure CAN transports are started:
```bash
source ~/Projects/Berkeley-Humanoid-Lite/scripts/start_can_transports.sh
```

#### Using C Codebase (Recommended for Locomotion)

The C codebase provides real-time control with better performance guarantees.

```bash
cd ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel
./build/main
```

**Expected behavior:**
- Robot starts in idle mode (motors relaxed)
- Gamepad controls robot movement (see Testing Procedures below)
- Policy runs at ~25 Hz, motor control at ~250 Hz

#### Using Python Codebase (Alternative)

The Python codebase is easier to modify but may have slightly lower performance.

```bash
cd ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel
uv run ./scripts/python/run_locomotion.py
```

**Note:** If you rebuild the C codebase after changes, run `make` first:
```bash
cd ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel
make
./build/main
```

## Testing Procedures

### Test Sequence

Follow this sequence to safely test the robot:

#### 1. Initial Position Test (Mode 1 - Idle)

- [ ] Robot starts in idle mode (motors relaxed)
- [ ] Press **X** button to ensure idle mode
- [ ] Verify robot doesn't make unexpected movements
- [ ] Check all joints are free to move manually

#### 2. Standing Pose Test (Mode 2 - Init)

- [ ] Press **A + L1** (Left Bumper) to enter init mode
- [ ] Robot should smoothly move to default standing pose
- [ ] Verify no jerky movements or oscillations
- [ ] Check robot balance and stability
- [ ] Observe torso orientation (should be upright)

#### 3. RL Control Test (Mode 3 - RL Control)

- [ ] Press **A + R1** (Right Bumper) to enter RL control mode
- [ ] **Do not move sticks yet** - verify robot maintains stance
- [ ] Slowly move **left stick forward** (small amount)
  - [ ] Robot should step forward smoothly
  - [ ] Verify gait is stable and coordinated
- [ ] Slowly move **left stick backward**
  - [ ] Robot should walk backward
- [ ] Test **left stick left/right** (turning)
  - [ ] Robot should turn in place or while walking
- [ ] Test **right stick left/right** (strafing)
  - [ ] Robot should step sideways

#### 4. Emergency Stop Test

- [ ] While robot is in RL control, press **X** to enter idle mode
- [ ] Robot should immediately stop and relax motors
- [ ] Verify emergency stop is responsive

### Data Collection

During testing, monitor and log:

- Motor temperatures
- Torque outputs (check against effort limits)
- IMU readings (angular velocity, orientation)
- Policy inference frequency (should be ~25 Hz)
- Motor control frequency (should be ~250 Hz)
- Gamepad command values
- Any error messages or warnings

### Visualization

On your development machine, visualize the robot state:

```bash
python scripts/sim2real/visualize.py
```

This receives UDP packets from the robot and displays the state in MuJoCo viewer.

## Open Items & Verification Needed

### Gamepad Controller Verification

**Status:** ⚠️ **REQUIRES REAL ROBOT TESTING**

**Recent Changes:**
On 2025-10-23, the gamepad controller module was significantly refactored to support multiple controller types:

1. **Multi-controller support** - Auto-detects PS5 DualSense vs XInput/Xbox controllers
2. **Fixed normalization** - PS5 uses 8-bit values (0-255), Xbox uses 16-bit (-32768 to 32767)
3. **Dead zone fix** - Dead zone filtering now actually applied (was previously defined but not used)
4. **Improved device detection** - Filters out keyboards with gamepad interfaces

**Backward Compatibility:**
- ✅ XInput/Xbox controllers: Mathematically identical normalization
- ✅ Dead zone now active: Prevents drift (this is a fix, not a regression)
- ✅ Auto-detection: Falls back to XInput profile if detection fails

**What to Verify on Real Robot:**

1. **Controller Detection**
   ```bash
   # Check which controller is detected
   python ./source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/policy/udp_joystick.py
   ```
   Expected output:
   ```
   Detected gamepad: [Your Controller Name]
   Using controller: XInput/Xbox Controller (or Sony DualSense)
   ```

2. **Movement Response**
   - [ ] Robot responds to small stick movements (dead zone ~1%)
   - [ ] Maximum stick deflection produces expected max velocity
   - [ ] Turning and strafing work as before
   - [ ] No unexpected sensitivity issues

3. **Mode Switching**
   - [ ] **A + R1** enters RL control mode
   - [ ] **A + L1** enters init mode
   - [ ] **X** enters idle mode
   - [ ] Button combinations register correctly

**If Issues Occur:**

If the robot doesn't respond correctly to gamepad inputs, add a legacy mode fallback:

#### Option 1: Quick Fix (Command Line)

Edit `source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/robot/humanoid.py`:

```python
# Find this line (around line 130):
self.command_controller = Se2Gamepad()

# Change to:
self.command_controller = Se2Gamepad(legacy_mode=True)
```

#### Option 2: Add Legacy Mode Flag

Add `legacy_mode` parameter to `Se2Gamepad` class in `gamepad.py`:

```python
class Se2Gamepad:
    def __init__(self,
                 stick_sensitivity: float = 1.0,
                 dead_zone: float = 0.01,
                 debug: bool = False,
                 legacy_mode: bool = False,  # Add this
                 ) -> None:
        self.stick_sensitivity = stick_sensitivity
        self.dead_zone = dead_zone
        self.debug = debug
        self.legacy_mode = legacy_mode

        # ... existing code ...

        if legacy_mode:
            # Force XInput profile, no auto-detection
            self._gamepad_device = None
            self._controller_profile = CONTROLLER_PROFILES['xinput']
            print("Using legacy mode: XInput profile (no auto-detection)")
        else:
            # Detect and configure controller
            self._gamepad_device, self._controller_profile = self._detect_controller()
            print(f"Using controller: {self._controller_profile.name}")
```

Then enable it in deployment:
```python
self.command_controller = Se2Gamepad(legacy_mode=True)
```

**Report Back:**
Please document your findings:
- Controller type used (make/model)
- Whether auto-detection worked correctly
- Any sensitivity or responsiveness issues
- Whether legacy_mode was needed

### Other Open Items

#### Motor Effort Limits

- [ ] Verify torque limits in `policy_latest.yaml` match robot capabilities
- [ ] Monitor peak torques during testing
- [ ] Adjust if motors overheat or reach limits

#### Policy Frequency

- [ ] Confirm policy runs at 25 Hz consistently
- [ ] Check for timing violations or delays
- [ ] Consider real-time kernel if needed: https://xanmod.org/

#### UDP Communication

- [ ] Verify UDP packets transmitted correctly
- [ ] Check for packet loss during operation
- [ ] Confirm visualization receives data

## Troubleshooting

### Issue: Robot Doesn't Respond to Gamepad

**Symptoms:**
- Stick movements don't affect robot
- Velocity commands remain at zero

**Checks:**
1. Verify gamepad is connected: `ls /dev/input/js*`
2. Test gamepad separately: `python ./source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/policy/udp_joystick.py`
3. Check robot is in RL control mode (mode 3)
4. Try legacy mode (see [Gamepad Verification](#gamepad-controller-verification) above)

### Issue: Robot Movements Too Sensitive/Insensitive

**Possible Causes:**
- Controller normalization incorrect for your gamepad
- Dead zone too large/small
- Stick sensitivity misconfigured

**Solutions:**
1. Enable debug mode to see raw values:
   ```python
   self.command_controller = Se2Gamepad(debug=True)
   ```
2. Adjust dead zone:
   ```python
   self.command_controller = Se2Gamepad(dead_zone=0.005)  # Lower = more sensitive
   ```
3. Adjust stick sensitivity:
   ```python
   self.command_controller = Se2Gamepad(stick_sensitivity=0.5)  # Lower = less sensitive
   ```

### Issue: Motors Not Responding

**Checks:**
1. CAN transports active: `ip link show can0`
2. Motor power supply connected and on
3. CAN bus termination resistors in place
4. Individual motor ping succeeds

### Issue: IMU Readings Incorrect

**Checks:**
1. IMU serial connection: `ls /dev/ttyUSB*` or `/dev/ttyACM*`
2. Correct baudrate: 460800 (default)
3. IMU calibration procedure run
4. Check for magnetic interference near IMU

### Issue: Robot Falls or Unstable

**Possible Causes:**
- Policy not suited for real robot dynamics
- Hardware calibration issues
- Joint offset errors
- IMU orientation incorrect

**Solutions:**
1. Return to sim2sim validation - verify policy works there
2. Check joint zero positions match URDF
3. Verify IMU mounting orientation
4. Consider retraining with more domain randomization

## Safety Protocols

### Before Every Test

1. **Clear Testing Area**
   - No obstacles within 2 meters
   - Soft landing surface (foam mats)
   - No bystanders within 3 meters

2. **Robot Inspection**
   - All bolts tight
   - No loose wires
   - Motors not overheated from previous tests
   - Battery charged

3. **Emergency Procedures**
   - Kill switch/tether accessible
   - Power disconnect ready
   - At least one person monitoring

### During Testing

1. **Start Conservative**
   - Begin with idle mode
   - Test standing pose before walking
   - Start with small stick movements

2. **Monitor Continuously**
   - Watch for unusual movements
   - Listen for motor grinding/clicking
   - Check motor temperatures

3. **Stop Immediately If:**
   - Robot loses balance
   - Unusual sounds or vibrations
   - Motors overheating
   - Oscillations or instability
   - Any hardware damage

### After Testing

1. **Return to Idle**
   - Press **X** to enter idle mode
   - Wait for motors to relax
   - Power off safely

2. **Inspection**
   - Check all joints for wear
   - Inspect motor temperatures
   - Look for loose connections
   - Document any issues

## Related Documentation

- [MuJoCo Simulation Guide](mujoco-simulation-guide.md) - Sim2sim validation
- [Training Guide](training-guide.md) - Train policies in Isaac Sim
- [Policies Guide](policies-guide.md) - Export and manage policies
- [On-board Computer Setup](.external-libs/berkeley-humanoid-lite-docs/docs/getting-started-with-software/the-on-board-computer.md)

## Changelog

### 2025-10-23
- Initial deployment testing guide created
- Added comprehensive on-board computer initial setup section
  - System dependencies installation (including Boost libraries)
  - uv package manager installation and usage
  - Repository cloning and submodule initialization
  - C codebase build instructions
  - Python environment setup with uv sync
  - CAN transport testing
- Fixed policy file deployment paths (to lowlevel submodule)
- Added detailed deployment steps with correct paths
- Added gamepad controller verification as open item
- Documented recent gamepad refactoring changes
- Added legacy_mode fallback instructions
- Included comprehensive safety protocols
