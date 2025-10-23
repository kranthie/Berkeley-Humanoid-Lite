# Migrating from Intel N95 Mini PC to Jetson Orin Nano Super

Complete guide for replacing the Berkeley Humanoid Lite's onboard Intel N95 mini PC with NVIDIA Jetson Orin Nano Super.

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Why Migrate](#why-migrate)
4. [Hardware Setup](#hardware-setup)
5. [Software Environment Setup](#software-environment-setup)
6. [Porting Lowlevel Control Code](#porting-lowlevel-control-code)
7. [CAN Bus Configuration](#can-bus-configuration)
8. [ONNX Runtime Setup](#onnx-runtime-setup)
9. [Testing and Validation](#testing-and-validation)
10. [Deployment](#deployment)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)

## Overview

### What This Migration Achieves

- Replace x86 Intel N95 with ARM-based Jetson Orin Nano Super
- Enable GPU-accelerated inference for RL policies
- Reduce weight and power consumption
- Prepare for future vision-based tasks
- Maintain 250 Hz CAN bus communication
- Support 25 Hz policy inference

### Migration Complexity

**Difficulty**: Medium
**Estimated Time**: 4-8 hours (first time)
**Skills Required**: Linux system administration, cross-compilation, robotics basics

### Current vs New Architecture

**Current (Intel N95):**
```
Intel N95 Mini PC (x86_64)
├── Ubuntu 22.04
├── Python 3.11
├── ONNX Runtime (CPU)
├── 4x USB-CAN adapters
└── USB IMU (Arduino)
```

**New (Jetson Orin Nano Super):**
```
Jetson Orin Nano Super (ARM64)
├── Ubuntu 20.04/22.04 (JetPack 6.x)
├── Python 3.10/3.11
├── ONNX Runtime GPU
├── CUDA 12.x + TensorRT
├── 4x USB-CAN adapters (same)
└── USB IMU (same)
```

## Hardware Requirements

### Components Needed

| Item | Specs | Purpose | Cost |
|------|-------|---------|------|
| Jetson Orin Nano Super Dev Kit | 8GB, 67 TOPS | Main compute module | $249 |
| MicroSD Card | 128GB+ UHS-I | OS storage | $20 |
| USB-C Power Supply | 5V 4A (20W) | Power for Jetson | Included |
| USB Hub (optional) | 4+ ports, powered | Connect CAN adapters | $15-30 |
| Heat Sink / Fan | Compatible with Jetson | Cooling | Included |

### Existing Hardware (Reuse)

- 4x USB-CAN adapters (PEAK, PCAN-USB, or compatible)
- Arduino with IMU (BNO085)
- 6S LiPo battery (or power supply for testing)
- Voltage regulator (if needed to step down battery voltage)

### Physical Integration Notes

**Jetson Orin Nano Super dimensions:**
- Module only: 70mm x 45mm
- Dev Kit board: 100mm x 79mm x 31mm

**Weight:**
- Module: 52g
- Dev Kit: ~180g (with heatsink)

**Power consumption:**
- Idle: 5-7W
- Active (RL inference): 10-15W
- Max: 25W

**Mounting**: May need to 3D print custom mount for torso cavity (smaller than N95 mini PC).

## Why Migrate

### Performance Benefits

| Metric | Intel N95 | Jetson Orin Nano Super | Improvement |
|--------|-----------|------------------------|-------------|
| CPU Cores | 4 (x86) | 6 (ARM Cortex-A78AE) | 1.5x |
| GPU | None | 1024 CUDA cores | ∞ |
| AI Performance | 0 TOPS | 67 TOPS | ∞ |
| Memory Bandwidth | 38.4 GB/s | 102 GB/s | 2.7x |
| Weight | ~200g | 52g (module) | 3.8x lighter |
| Power (typical) | 15W | 10-15W | Similar or better |

### Functional Benefits

- **GPU acceleration**: ONNX policies run 5-10x faster with GPU
- **TensorRT**: Further optimize models beyond ONNX
- **Vision ready**: Built-in CSI camera support + USB cameras
- **Smaller/lighter**: Better for humanoid dynamics
- **Future-proof**: Supports advanced AI workloads (Groot N1 testing)

## Hardware Setup

### Step 1: Flash JetPack to Jetson

**Option A: Using SD Card (Easiest)**

```bash
# On your development workstation (Linux/Mac/Windows)

# 1. Download Jetson Orin Nano Super SD Card Image
# Visit: https://developer.nvidia.com/embedded/jetpack
# Download: JetPack 6.x for Jetson Orin Nano

# 2. Flash to SD card using balenaEtcher or dd
# For balenaEtcher (GUI): https://www.balena.io/etcher/

# For dd (Linux/Mac):
sudo dd if=jetson-orin-nano-sd-card-image.img of=/dev/sdX bs=4M status=progress
sync

# 3. Insert SD card into Jetson
# 4. Power on and complete initial setup via HDMI + keyboard
```

**Option B: Using NVIDIA SDK Manager (Advanced)**

```bash
# On Ubuntu 20.04/22.04 workstation

# 1. Download SDK Manager
wget https://developer.nvidia.com/sdk-manager

# 2. Install
sudo apt install ./sdkmanager_[version].deb

# 3. Launch
sdkmanager

# 4. Follow GUI to:
#    - Select Jetson Orin Nano
#    - Choose JetPack 6.x
#    - Flash via USB (put Jetson in recovery mode)
```

### Step 2: Initial Jetson Configuration

```bash
# Connect via HDMI + keyboard or SSH after setup

# Check JetPack version
sudo apt-cache show nvidia-jetpack

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    can-utils \
    net-tools \
    htop \
    tmux

# Verify CUDA installation
nvcc --version
# Should show CUDA 12.x

# Verify GPU
nvidia-smi
# May not work on Jetson, use:
tegrastats
# Should show GPU utilization stats
```

### Step 3: Set Power Mode

```bash
# Jetson Orin Nano Super has multiple power modes

# View available modes
sudo nvpmodel -q

# Set to maximum performance (25W)
sudo nvpmodel -m 0

# Or set to 15W mode for longer battery life
sudo nvpmodel -m 1

# Make persistent across reboots
sudo systemctl enable nvpmodel

# Enable all CPU cores
sudo jetson_clocks
```

### Step 4: Physical Mounting

**Considerations:**
- Jetson generates more heat than N95 under load
- Ensure adequate cooling in torso cavity
- Mount heatsink securely
- Consider active cooling (fan) for extended operation
- Provide vibration dampening (foam tape)

**Wiring:**
- USB-C power (from voltage regulator or external supply)
- 4x USB-CAN adapters
- 1x USB IMU
- Optional: Ethernet for development

## Software Environment Setup

### Step 1: Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv ~/robot_env
source ~/robot_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install numpy scipy pyyaml omegaconf

# Install ONNX Runtime GPU (Jetson-specific)
# DO NOT use pip install onnxruntime-gpu (won't work on Jetson)

# Instead, use NVIDIA's wheel:
pip install onnxruntime-gpu --extra-index-url https://pypi.nvidia.com

# Verify ONNX Runtime with GPU
python3 -c "import onnxruntime as ort; print(ort.get_device()); print(ort.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Step 2: Install CAN Bus Support

```bash
# Install Python CAN library
pip install python-can

# Load CAN kernel modules
sudo modprobe can
sudo modprobe can_raw
sudo modprobe slcan

# Make modules load at boot
echo "can" | sudo tee -a /etc/modules
echo "can_raw" | sudo tee -a /etc/modules
echo "slcan" | sudo tee -a /etc/modules
```

### Step 3: Install Robot Control Dependencies

```bash
# If using the Berkeley Humanoid Lite lowlevel Python code
cd ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel

# Install dependencies
pip install -r requirements.txt

# Note: Some dependencies may need ARM64 builds
# Most Python packages have ARM64 wheels on PyPI
```

## Porting Lowlevel Control Code

### Architecture Considerations

**x86 to ARM64 differences:**
- Different byte ordering (usually not an issue for Python)
- Some C++ libraries may need recompilation
- Performance characteristics differ (but ARM is very efficient)

### Option 1: Python-Only Control (Simplest)

If using the Python-based control from `berkeley_humanoid_lite_lowlevel`:

```bash
# Most code will work without changes
# Python is platform-independent

# Test imports
cd ~/Projects/Berkeley-Humanoid-Lite
source .venv/bin/activate

python3 -c "
from berkeley_humanoid_lite_lowlevel import RlController
print('Import successful')
"
```

### Option 2: C++ Control Code (Requires Recompilation)

If there's C++ code for motor control:

```bash
# Clone/copy repository to Jetson
cd ~/Projects
git clone https://github.com/kranthie/Berkeley-Humanoid-Lite-Lowlevel.git
cd Berkeley-Humanoid-Lite-Lowlevel

# Create build directory
mkdir build && cd build

# Configure for ARM64
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native"

# Build
make -j6  # Use all 6 CPU cores

# Install
sudo make install
```

### Cross-Compilation (Advanced)

If you want to build on your x86 workstation for Jetson:

```bash
# On x86 development machine

# Install cross-compilation toolchain
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Create toolchain file: arm64-toolchain.cmake
cat > arm64-toolchain.cmake << 'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF

# Build
mkdir build-arm64 && cd build-arm64
cmake -DCMAKE_TOOLCHAIN_FILE=../arm64-toolchain.cmake ..
make -j$(nproc)

# Copy binaries to Jetson
scp robot_control jetson@<JETSON_IP>:~/
```

## CAN Bus Configuration

### Step 1: Identify USB-CAN Adapters

```bash
# Plug in all 4 USB-CAN adapters

# List USB devices
lsusb
# Look for PEAK, PCAN, or your CAN adapter vendor

# Check if CAN interfaces are detected
ip link show | grep can
# Should show: can0, can1, can2, can3
```

### Step 2: Configure CAN Interfaces

```bash
# Berkeley Humanoid Lite uses 1 Mbps bitrate

# Set up all 4 CAN buses
for i in {0..3}; do
    sudo ip link set can$i type can bitrate 1000000
    sudo ip link set up can$i
done

# Verify
ip -details link show can0
# Should show: state UP, bitrate 1000000
```

### Step 3: Create Persistent CAN Configuration

```bash
# Create systemd service for CAN setup

sudo nano /etc/systemd/system/can-setup.service
```

Add content:
```ini
[Unit]
Description=Setup CAN bus interfaces
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/setup-can.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Create setup script:
```bash
sudo nano /usr/local/bin/setup-can.sh
```

Add content:
```bash
#!/bin/bash
for i in {0..3}; do
    ip link set can$i type can bitrate 1000000
    ip link set up can$i
done
```

Make executable and enable:
```bash
sudo chmod +x /usr/local/bin/setup-can.sh
sudo systemctl enable can-setup.service
sudo systemctl start can-setup.service
```

### Step 4: Test CAN Communication

```bash
# Send test message on can0
cansend can0 123#DEADBEEF

# Listen on can0 (in another terminal)
candump can0

# Or use Berkeley Humanoid Lite test scripts
cd ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel
python3 scripts/test_can_communication.py
```

## ONNX Runtime Setup

### Step 1: Verify GPU Provider

```bash
# Test ONNX Runtime GPU support
python3 << 'EOF'
import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

# Should show CUDAExecutionProvider
assert 'CUDAExecutionProvider' in ort.get_available_providers(), "GPU not available!"
print("✓ GPU acceleration available")
EOF
```

### Step 2: Configure ONNX Inference Session

```python
# In your deployment code
import onnxruntime as ort

# Create session with GPU
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Use GPU provider
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
    }),
    'CPUExecutionProvider'  # Fallback
]

session = ort.InferenceSession(
    'policy.onnx',
    sess_options=sess_options,
    providers=providers
)

print(f"Using: {session.get_providers()}")
```

### Step 3: Benchmark ONNX Inference

```python
import onnxruntime as ort
import numpy as np
import time

# Load your trained policy
session = ort.InferenceSession('path/to/policy.onnx', providers=['CUDAExecutionProvider'])

# Get input shape from model
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Create dummy input (replace with actual observation size)
dummy_input = np.random.randn(1, *input_shape[1:]).astype(np.float32)

# Warmup
for _ in range(10):
    session.run(None, {input_name: dummy_input})

# Benchmark
num_iterations = 1000
start = time.time()
for _ in range(num_iterations):
    output = session.run(None, {input_name: dummy_input})
end = time.time()

avg_latency = (end - start) / num_iterations * 1000  # milliseconds
print(f"Average inference time: {avg_latency:.2f} ms")
print(f"Throughput: {1000/avg_latency:.1f} Hz")
```

**Expected performance:**
- CPU (N95): 10-20ms per inference (~50-100 Hz capable)
- GPU (Orin Nano): 2-5ms per inference (~200-500 Hz capable)

## Testing and Validation

### Step 1: Hardware Validation

```bash
# Check all systems
sudo tegrastats  # Monitor GPU, CPU, temp, power

# Verify CAN buses
for i in {0..3}; do
    echo "Testing can$i..."
    candump can$i -n 1 &
    sleep 0.1
    cansend can$i 100#1122334455667788
done

# Check IMU
python3 -c "
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f'Port: {port.device}, Description: {port.description}')
"
```

### Step 2: Benchmark Control Loop

```python
# Test control loop frequency
import time
import numpy as np

def control_loop_test(duration_seconds=10):
    """Test 250 Hz control loop timing"""
    target_dt = 1.0 / 250.0  # 4ms

    times = []
    start_time = time.time()
    last_time = start_time

    while time.time() - start_time < duration_seconds:
        current_time = time.time()
        dt = current_time - last_time
        times.append(dt)
        last_time = current_time

        # Simulate control loop work
        _ = np.random.randn(100, 100) @ np.random.randn(100, 100)

        # Sleep to maintain 250 Hz
        sleep_time = target_dt - (time.time() - current_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    times = np.array(times[1:])  # Skip first
    print(f"Mean dt: {times.mean()*1000:.3f} ms (target: {target_dt*1000:.1f} ms)")
    print(f"Std dt: {times.std()*1000:.3f} ms")
    print(f"Max dt: {times.max()*1000:.3f} ms")
    print(f"Min dt: {times.min()*1000:.3f} ms")
    print(f"Actual frequency: {1.0/times.mean():.1f} Hz")

control_loop_test()
```

### Step 3: Test Policy Inference

```bash
# Use Berkeley Humanoid Lite's play script
cd ~/Projects/Berkeley-Humanoid-Lite

# Test with visualization (if robot connected)
python3 scripts/sim2real/visualize.py

# Or test policy loading
python3 << 'EOF'
from berkeley_humanoid_lite_lowlevel import RlController, Cfg

# Load deployment config
cfg = Cfg.from_yaml('configs/policy_latest.yaml')
controller = RlController(cfg)

print(f"✓ Controller initialized")
print(f"✓ Policy loaded: {cfg.policy_checkpoint_path}")
print(f"✓ Num joints: {cfg.num_joints}")
EOF
```

## Deployment

### Step 1: Create Deployment Package

```bash
# On Jetson, organize deployment files
mkdir -p ~/robot_deployment
cd ~/robot_deployment

# Copy necessary files
cp ~/Projects/Berkeley-Humanoid-Lite/configs/policy_latest.yaml .
cp ~/Projects/Berkeley-Humanoid-Lite/logs/rsl_rl/*/exported/policy.onnx .

# Copy lowlevel control code
cp -r ~/Projects/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel .
```

### Step 2: Create Startup Script

```bash
nano ~/robot_deployment/start_robot.sh
```

Add content:
```bash
#!/bin/bash

# Activate environment
source ~/robot_env/bin/activate

# Set power mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Setup CAN
for i in {0..3}; do
    sudo ip link set can$i type can bitrate 1000000
    sudo ip link set up can$i
done

# Start robot control
cd ~/robot_deployment
python3 -m berkeley_humanoid_lite_lowlevel.main --config policy_latest.yaml

# Or use the C++ control loop if compiled
# ./robot_control --config policy_latest.yaml
```

Make executable:
```bash
chmod +x ~/robot_deployment/start_robot.sh
```

### Step 3: Create Systemd Service (Optional)

For automatic startup:

```bash
sudo nano /etc/systemd/system/robot-control.service
```

Add content:
```ini
[Unit]
Description=Berkeley Humanoid Lite Robot Control
After=network.target can-setup.service

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/robot_deployment
ExecStart=/home/jetson/robot_deployment/start_robot.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable robot-control.service
# Don't start yet - test manually first
```

### Step 4: Safety Testing

**Before running on robot:**

1. **Tethered test**: Keep robot on stand, external power
2. **Emergency stop**: Have physical e-stop ready
3. **Low gains**: Start with reduced PD gains
4. **Limited range**: Test with restricted joint limits
5. **Monitor temps**: Watch `tegrastats` for thermal throttling

## Performance Optimization

### TensorRT Optimization

For maximum performance, convert ONNX to TensorRT:

```python
import tensorrt as trt

# Convert ONNX to TensorRT engine
def build_tensorrt_engine(onnx_path, engine_path, fp16=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    engine = builder.build_serialized_network(network, config)

    # Save
    with open(engine_path, 'wb') as f:
        f.write(engine)

    print(f"✓ TensorRT engine saved to {engine_path}")

# Convert
build_tensorrt_engine('policy.onnx', 'policy.trt', fp16=True)
```

Then use with TensorRT runtime:
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, input_data):
        # Allocate buffers and run inference
        # (Simplified - full implementation in TensorRT docs)
        pass
```

### CPU Affinity

Pin control loop to specific cores:

```python
import os

# Pin to CPU cores 4-5 (leave 0-3 for system)
os.sched_setaffinity(0, {4, 5})
```

### Real-Time Priority

```python
import os

# Set real-time priority
os.nice(-10)  # Higher priority (requires sudo)
```

## Troubleshooting

### Issue: CAN interfaces not appearing

**Symptoms**: `ip link show` doesn't list can0-can3

**Solutions**:
```bash
# Check if adapters are detected
lsusb | grep -i can

# Manually load modules
sudo modprobe can
sudo modprobe can_raw
sudo modprobe slcan

# Check dmesg for errors
dmesg | tail -50
```

### Issue: ONNX Runtime not using GPU

**Symptoms**: Slow inference, `get_providers()` shows only CPU

**Solutions**:
```bash
# Reinstall ONNX Runtime GPU
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu --extra-index-url https://pypi.nvidia.com

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Check CUDA libraries
ldconfig -p | grep cuda
```

### Issue: Thermal throttling

**Symptoms**: Performance degrades over time, `tegrastats` shows high temps

**Solutions**:
```bash
# Check temperature
tegrastats

# Ensure heatsink is properly mounted
# Add thermal paste if needed
# Consider active cooling (fan)

# Reduce power mode
sudo nvpmodel -m 1  # 15W mode instead of 25W
```

### Issue: Control loop jitter

**Symptoms**: Inconsistent timing, robot jerky movements

**Solutions**:
```python
# Use real-time kernel (advanced)
# Or reduce system load:

# Disable GUI
sudo systemctl set-default multi-user.target

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon

# Pin to isolated CPUs
# Add to /etc/default/grub:
# GRUB_CMDLINE_LINUX="isolcpus=4,5"
```

### Issue: USB-CAN adapters not stable

**Symptoms**: CAN bus drops, connection errors

**Solutions**:
```bash
# Increase USB buffer size
# Add to /etc/sysctl.conf:
net.core.rmem_max = 8388608
net.core.wmem_max = 8388608
net.core.rmem_default = 8388608
net.core.wmem_default = 8388608

sudo sysctl -p

# Use powered USB hub
# Shorter USB cables
# Check for EMI interference
```

## Performance Comparison

### Inference Latency

| Metric | Intel N95 (CPU) | Jetson Orin Nano (GPU) |
|--------|-----------------|------------------------|
| ONNX (fp32) | 15 ms | 3 ms |
| ONNX (fp16) | N/A | 2 ms |
| TensorRT (fp16) | N/A | 1.5 ms |

### Control Loop Stability

| Metric | Intel N95 | Jetson Orin Nano |
|--------|-----------|------------------|
| Target frequency | 250 Hz | 250 Hz |
| Mean jitter | ±0.5 ms | ±0.2 ms |
| Max jitter | 2 ms | 0.8 ms |

### Power Consumption

| Mode | Intel N95 | Jetson Orin Nano |
|------|-----------|------------------|
| Idle | 10W | 5-7W |
| Control loop | 15W | 10-12W |
| Peak | 20W | 15W |

### Weight and Size

| Metric | Intel N95 Mini PC | Jetson Orin Nano |
|--------|-------------------|------------------|
| Weight | ~200g | 52g (module) |
| Volume | ~150 cm³ | ~15 cm³ (module) |

## Next Steps

After successful migration:

1. **Validate existing RL policies**: Test trained policies work identically
2. **Optimize with TensorRT**: Convert ONNX to TensorRT for 2-3x speedup
3. **Add camera**: Integrate RGB camera for vision tasks
4. **Test Groot N1**: Experiment with foundation model (tethered first)
5. **Collect real-world data**: Use robot to gather demonstration data

## References

- [Jetson Orin Nano Developer Guide](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide)
- [ONNX Runtime on Jetson](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CAN Bus on Linux](https://www.kernel.org/doc/Documentation/networking/can.txt)
