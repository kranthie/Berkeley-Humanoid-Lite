# Policy Development Guide

Comprehensive guide to creating and customizing policies for Berkeley Humanoid Lite using Isaac Lab's Manager-Based RL framework.

## Table of Contents

1. [Understanding the Current Policy](#understanding-the-current-policy)
2. [Modifying Reward Functions](#modifying-reward-functions)
3. [Creating Custom Rewards](#creating-custom-rewards)
4. [Modifying Observations and Actions](#modifying-observations-and-actions)
5. [Creating New Tasks](#creating-new-tasks)
6. [Multi-Objective Policies](#multi-objective-policies)
7. [Curriculum Learning](#curriculum-learning)
8. [Examples](#examples)

## Understanding the Current Policy

The existing `Velocity-Berkeley-Humanoid-Lite-v0` task trains the robot to walk at commanded velocities.

### What It Does

- **Input**: Velocity commands (linear x/y, angular z)
- **Output**: Joint position targets for all 22 joints
- **Goal**: Track velocity commands while maintaining balance and stability

### MDP Components

All tasks are defined using Isaac Lab's Manager-Based structure:

```
source/berkeley_humanoid_lite/tasks/locomotion/velocity/config/humanoid/
├── __init__.py
├── env_cfg.py           # Main environment configuration
└── agents/
    └── rsl_rl_ppo_cfg.py  # PPO hyperparameters
```

**Key classes in `env_cfg.py`**:

```python
@configclass
class BerkeleyHumanoidLiteEnvCfg(LocomotionVelocityEnvCfg):
    commands: CommandsCfg           # What the robot tries to achieve
    observations: ObservationsCfg   # What the policy sees
    actions: ActionsCfg             # What the policy controls
    rewards: RewardsCfg             # What behaviors are encouraged/discouraged
    terminations: TerminationsCfg   # When episodes end
    events: EventsCfg               # Randomization and domain randomization
    curriculums: CurriculumsCfg     # Progressive difficulty
```

## Modifying Reward Functions

Rewards shape what the policy learns. The framework makes this easy.

### Current Reward Structure

Located in `env_cfg.py:105-215`:

```python
@configclass
class RewardsCfg:
    # Task performance
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,  # Higher weight = more important
    )

    # Penalties
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,  # Negative = penalty
    )

    # Smoothness
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,  # Small penalty for jerky movements
    )
```

### Tuning Weights

**To emphasize different behaviors**, just change weights:

```python
# Make velocity tracking more important
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_yaw_frame_exp,
    weight=5.0,  # Increased from 2.0
)

# Penalize energy use more heavily
dof_torques_l2 = RewTerm(
    func=mdp.joint_torques_l2,
    weight=-1.0e-4,  # Increased from -2.0e-5
)
```

**Guidelines**:
- Positive weights = encourage behavior
- Negative weights = discourage behavior
- Magnitude matters: 2.0 vs 0.001 is very different
- Balance is key: one dominant reward can ignore others
- Monitor in TensorBoard under "Rewards/" to see individual contributions

### Adding/Removing Rewards

**Remove a reward**: Just delete or comment out:

```python
# Don't care about feet air time anymore
# feet_air_time = RewTerm(...)
```

**Add an existing reward**: Browse available functions in:
- `source/berkeley_humanoid_lite/tasks/locomotion/velocity/mdp/rewards.py`
- Isaac Lab's `isaaclab.envs.mdp` module

Example - add foot clearance reward:

```python
foot_clearance = RewTerm(
    func=mdp.foot_clearance,  # Reward lifting feet high
    params={"threshold": 0.1},  # 10cm minimum height
    weight=0.5,
)
```

## Creating Custom Rewards

When built-in rewards aren't enough, write your own.

### Basic Custom Reward

**File**: `source/berkeley_humanoid_lite/tasks/locomotion/velocity/mdp/rewards.py`

```python
def my_custom_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for some custom behavior.

    Args:
        env: The RL environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Access robot state
    robot = env.scene["robot"]

    # Example: Reward keeping base height at 0.8m
    base_height = robot.data.root_pos_w[:, 2]  # Z position
    target_height = 0.8

    # Exponential kernel (smooth, differentiable)
    error = torch.abs(base_height - target_height)
    reward = torch.exp(-error / 0.1**2)

    return reward
```

**Register in env_cfg.py**:

```python
from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp

@configclass
class RewardsCfg:
    # ... existing rewards ...

    maintain_height = RewTerm(
        func=mdp.my_custom_reward,
        weight=1.0,
    )
```

### Reward with Parameters

More flexible rewards accept parameters:

```python
def joint_position_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_positions: list[float],
    std: float
) -> torch.Tensor:
    """Reward for tracking specific joint positions.

    Args:
        env: Environment
        asset_cfg: Which robot/joints to track
        target_positions: Desired joint positions
        std: Standard deviation for exponential kernel
    """
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]

    targets = torch.tensor(target_positions, device=env.device)
    error = torch.sum((joint_pos - targets)**2, dim=1)

    return torch.exp(-error / std**2)
```

**Use it**:

```python
arm_pose_tracking = RewTerm(
    func=mdp.joint_position_tracking,
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"]),
        "target_positions": [0.5, -0.3, 0.1, -0.2, 0.5, -0.3, 0.1, -0.2],  # 8 arm joints
        "std": 0.1,
    },
    weight=2.0,
)
```

### Common Reward Patterns

**Exponential kernel** (smooth, always positive):
```python
reward = torch.exp(-error / std**2)
```

**L2 penalty** (grows quadratically):
```python
penalty = torch.sum(values**2, dim=1)
```

**L1 penalty** (linear growth):
```python
penalty = torch.sum(torch.abs(values), dim=1)
```

**Threshold-based** (binary):
```python
reward = (values > threshold).float()
```

**Conditional rewards**:
```python
# Only reward when moving
is_moving = torch.norm(command[:, :2], dim=1) > 0.1
reward = base_reward * is_moving
```

## Modifying Observations and Actions

### Observations

What the policy "sees". Current observations (env_cfg.py:43-88):

```python
@configclass
class PolicyCfg(ObsGroup):
    """What the neural network receives as input."""

    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"}
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-0.3, n_max=0.3),  # Add noise for robustness
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_JOINTS)},
    )
    # ... more observations
```

**Add a new observation**:

```python
contact_forces = ObsTerm(
    func=mdp.contact_forces,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll")
    },
)
```

**Observation groups**:
- `policy`: What the actor network sees (limited, noisy for sim2real)
- `critic`: What the value network sees (can have privileged info like true velocities)

### Actions

What the policy controls. Current setup (env_cfg.py:92-101):

```python
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HUMANOID_LITE_JOINTS,  # All 22 joints
        scale=0.25,  # Action is added to default pose with scaling
        use_default_offset=True,
    )
```

**Action types**:
- `JointPositionActionCfg` - Target positions (PD controller)
- `JointVelocityActionCfg` - Target velocities
- `JointEffortActionCfg` - Direct torque control

**Modify action space**:

```python
# Use larger action scaling for more aggressive movements
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=HUMANOID_LITE_JOINTS,
    scale=0.5,  # Increased from 0.25
)

# Or control subset of joints
leg_joints_only = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],  # Legs only
    scale=0.25,
)
```

## Creating New Tasks

To create a completely new task (e.g., jumping, object manipulation):

### 1. Create Task Directory

```
source/berkeley_humanoid_lite/tasks/locomotion/
└── jumping/                          # New task
    ├── __init__.py
    ├── jumping_env_cfg.py            # Environment config
    ├── config/
    │   └── humanoid/
    │       ├── __init__.py
    │       ├── env_cfg.py            # Full config with rewards
    │       └── agents/
    │           └── rsl_rl_ppo_cfg.py
    └── mdp/                          # Custom MDP terms
        ├── __init__.py
        ├── rewards.py
        ├── observations.py
        └── commands.py
```

### 2. Define Base Environment

**jumping_env_cfg.py**:

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

@configclass
class JumpingEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for jumping task."""

    # Inherit scene from locomotion velocity
    scene: FlatTerrainSceneCfg = FlatTerrainSceneCfg(num_envs=4096)

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0  # Shorter episodes for jumping
        self.sim.dt = 0.005
```

### 3. Define MDP Components

**config/humanoid/env_cfg.py**:

```python
@configclass
class CommandsCfg:
    """Jump at specified times."""
    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(2.0, 5.0),  # New jump every 2-5 seconds
        height_range=(0.2, 0.6),  # Jump height in meters
    )

@configclass
class RewardsCfg:
    """Jumping-specific rewards."""

    # Reward achieving target jump height
    jump_height = RewTerm(
        func=mdp.track_jump_height,
        params={"command_name": "jump_command"},
        weight=5.0,
    )

    # Reward landing stability
    landing_stability = RewTerm(
        func=mdp.landing_stability,
        weight=2.0,
    )

    # Penalize falling
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )

    # Energy efficiency
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
    )

@configclass
class JumpingHumanoidEnvCfg(JumpingEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()  # Define your observations
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
```

### 4. Implement Custom MDP Terms

**mdp/rewards.py**:

```python
def track_jump_height(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward for achieving commanded jump height."""
    robot = env.scene["robot"]

    # Get current height
    current_height = robot.data.root_pos_w[:, 2]

    # Get commanded height
    target_height = env.command_manager.get_command(command_name)[:, 0]

    # Reward when height matches command
    error = torch.abs(current_height - target_height)
    return torch.exp(-error / 0.1**2)

def landing_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward stable landing after jump."""
    robot = env.scene["robot"]

    # Low angular velocity = stable
    ang_vel = robot.data.root_ang_vel_w
    stability = torch.exp(-torch.norm(ang_vel, dim=1) / 0.5**2)

    # Only reward when close to ground
    height = robot.data.root_pos_w[:, 2]
    near_ground = (height < 0.85).float()  # Assuming standing height ~0.8m

    return stability * near_ground
```

**mdp/commands.py**:

```python
@configclass
class JumpCommandCfg:
    """Configuration for jump commands."""
    resampling_time_range: tuple[float, float] = (2.0, 5.0)
    height_range: tuple[float, float] = (0.2, 0.6)

    # Implementation would generate random jump commands
```

### 5. Register Task

**config/humanoid/__init__.py**:

```python
import gymnasium as gym

from . import agents
from .env_cfg import JumpingHumanoidEnvCfg

gym.register(
    id="Jumping-Berkeley-Humanoid-Lite-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JumpingHumanoidEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.JumpingPPORunnerCfg,
    },
)
```

### 6. Train New Task

```bash
uv run ./scripts/rsl_rl/train.py \
    --task Jumping-Berkeley-Humanoid-Lite-v0 \
    --num_envs 4096 \
    --headless
```

## Multi-Objective Policies

A single policy can handle multiple objectives by combining rewards.

### Approach 1: Weighted Sum (Current Method)

The velocity-tracking policy already does this:

```python
@configclass
class RewardsCfg:
    # Objective 1: Track velocity
    track_lin_vel_xy_exp = RewTerm(func=..., weight=2.0)
    track_ang_vel_z_exp = RewTerm(func=..., weight=1.0)

    # Objective 2: Stability
    flat_orientation_l2 = RewTerm(func=..., weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=..., weight=-0.1)

    # Objective 3: Smoothness
    action_rate_l2 = RewTerm(func=..., weight=-0.001)

    # Objective 4: Natural gait
    feet_air_time = RewTerm(func=..., weight=2.0)
```

Total reward = weighted sum of all terms. Policy learns to balance all objectives.

**Tuning**: Adjust weights to prioritize objectives. Use TensorBoard to monitor individual reward contributions.

### Approach 2: Task Conditioning

Train one policy for multiple distinct tasks using task IDs.

**Add task ID to observations**:

```python
@configclass
class ObservationsCfg:
    task_id = ObsTerm(
        func=mdp.get_task_id,  # Returns one-hot encoded task [1,0,0] or [0,1,0] etc
    )
    # ... other observations
```

**Randomize task per environment**:

```python
@configclass
class EventsCfg:
    randomize_task = EventTerm(
        func=mdp.randomize_task_id,
        mode="reset",  # New task each episode
    )
```

**Use conditional rewards**:

```python
def conditional_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Different reward based on task."""
    task_id = env.task_id  # 0, 1, or 2

    base_reward = compute_base_reward(env)

    # Task 0: Walking
    # Task 1: Running
    # Task 2: Standing still

    reward = torch.where(
        task_id == 0,
        base_reward * 1.0,  # Walking weight
        torch.where(
            task_id == 1,
            base_reward * 2.0,  # Running weight
            base_reward * 0.5   # Standing weight
        )
    )
    return reward
```

### Approach 3: Hierarchical Policies

Use separate high-level and low-level policies (requires custom implementation).

**High-level policy**:
- Decides what primitive to execute (walk, jump, crouch)
- Runs at low frequency (1-5 Hz)
- Outputs: primitive ID + parameters

**Low-level policy**:
- Executes the primitive
- Runs at normal frequency (25 Hz)
- Outputs: joint targets

This requires significant custom code beyond the manager-based framework.

## Curriculum Learning

Gradually increase task difficulty during training.

### Built-in Curriculum Support

Isaac Lab provides `CurriculumsCfg`:

```python
@configclass
class CurriculumsCfg:
    """Progressive difficulty."""

    terrain_difficulty = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={
            "threshold": 0.5,  # Advance when mean reward > 0.5
            "initial_level": 0,
            "max_level": 10,
        }
    )
```

### Command Curriculum

Start with easy commands, progress to harder ones:

```python
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),  # Start slow
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
    )

@configclass
class CurriculumsCfg:
    velocity_difficulty = CurrTerm(
        func=mdp.increase_velocity_range,
        params={
            "threshold": 1.0,  # Advance when comfortable
            "max_lin_vel": 2.0,  # Eventually reach 2 m/s
            "max_ang_vel": 2.0,
        }
    )
```

Implement curriculum function:

```python
def increase_velocity_range(env, threshold: float, max_lin_vel: float, max_ang_vel: float):
    """Gradually increase velocity command ranges."""
    # Check if policy is doing well
    if env.episode_reward_mean > threshold:
        # Increase command ranges
        current_max = env.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x[1]
        new_max = min(current_max * 1.1, max_lin_vel)

        env.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x = (-new_max, new_max)
```

### Domain Randomization Curriculum

Start with accurate sim, add noise gradually:

```python
@configclass
class EventsCfg:
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "mass_distribution_params": (-0.5, 0.5),  # Start with small range
        },
        mode="startup",
    )

@configclass
class CurriculumsCfg:
    domain_rand_difficulty = CurrTerm(
        func=mdp.increase_domain_randomization,
        params={"threshold": 1.5, "max_mass_range": 3.0},
    )
```

## Examples

### Example 1: Energy-Efficient Walking

Modify the existing velocity task to prioritize energy efficiency:

```python
@configclass
class RewardsCfg:
    # Keep existing velocity tracking
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0)

    # Significantly increase torque penalty
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-3,  # 50x increase from -2.0e-5
    )

    # Add power consumption penalty
    power_consumption = RewTerm(
        func=mdp.power_consumption,  # torque * velocity
        weight=-5.0e-4,
    )

    # Reward slower, smoother movements
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,  # 10x increase from -0.001
    )
```

### Example 2: Arm Gesturing While Walking

Add arm control to the walking task:

```python
# New reward in mdp/rewards.py
def arm_wave_tracking(env: ManagerBasedRLEnv, frequency: float) -> torch.Tensor:
    """Reward arms waving at specified frequency."""
    robot = env.scene["robot"]

    # Get arm joint positions (shoulder pitch)
    left_shoulder = robot.data.joint_pos[:, robot.joint_names.index("arm_left_shoulder_pitch_joint")]
    right_shoulder = robot.data.joint_pos[:, robot.joint_names.index("arm_right_shoulder_pitch_joint")]

    # Target: sinusoidal wave
    time = env.episode_length_buf * env.step_dt
    target_pos = 0.3 * torch.sin(2 * torch.pi * frequency * time)

    error_left = torch.abs(left_shoulder - target_pos)
    error_right = torch.abs(right_shoulder + target_pos)  # Opposite phase

    return torch.exp(-(error_left + error_right) / 0.1**2)

# In env_cfg.py
@configclass
class RewardsCfg:
    # ... existing rewards ...

    arm_waving = RewTerm(
        func=mdp.arm_wave_tracking,
        params={"frequency": 0.5},  # 0.5 Hz waving
        weight=1.0,
    )
```

### Example 3: Backwards Walking Specialization

Create a policy that only walks backwards (useful for specialized behaviors):

```python
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, -0.2),  # Only negative (backward) velocities
            lin_vel_y=(-0.3, 0.3),   # Still allow lateral
            ang_vel_z=(-1.0, 1.0),
        ),
    )
```

## Best Practices

### Reward Design

1. **Start simple** - Get basic behavior working before adding complexity
2. **Monitor individual rewards** - Use TensorBoard to see contribution of each term
3. **Balance scales** - Ensure no single reward dominates (check magnitudes in TB)
4. **Use shaped rewards** - Exponential kernels work better than sparse binary rewards
5. **Avoid reward hacking** - Watch for unexpected behaviors that maximize reward

### Debugging

1. **Reduce num_envs** - Use 16-64 envs for faster iteration when debugging
2. **Disable randomization** - Comment out `EventsCfg` to remove noise
3. **Visualize** - Remove `--headless` to see what the robot is actually doing
4. **Log everything** - Add custom TensorBoard logs for debugging
5. **Check terminations** - Ensure episodes aren't ending too early

### Iteration Workflow

1. Modify reward weights → Train for 500-1000 iterations → Check TensorBoard
2. Add new reward term → Train briefly → Verify it's activating correctly
3. Adjust observation noise → Test in sim2sim to verify robustness
4. Tune PPO hyperparameters (in agents/rsl_rl_ppo_cfg.py) if learning is unstable

### Performance

- More complex reward functions slow down training (keep them simple)
- More observations increase network size and sample complexity
- Curriculum learning can significantly speed up training for complex tasks
- Domain randomization is essential for sim2real but slows learning

## Resources

- **Isaac Lab Docs**: https://isaac-sim.github.io/IsaacLab/
- **Isaac Lab MDP Terms**: Check `isaaclab.envs.mdp` for built-in functions
- **This Codebase**: Browse `mdp/rewards.py` for examples
- **Paper**: Berkeley Humanoid Lite paper for training details

## Summary

To create/modify policies:

1. **Modify existing**: Just adjust weights in `RewardsCfg`
2. **Add custom rewards**: Write function in `mdp/rewards.py`, register in `RewardsCfg`
3. **Change obs/actions**: Modify `ObservationsCfg` and `ActionsCfg`
4. **New task**: Create new directory structure, define all MDP components, register with gym
5. **Multi-objective**: Combine multiple rewards with appropriate weights
6. **Curriculum**: Use `CurriculumsCfg` to progressively increase difficulty

The framework is designed to be modular and extensible - start with the existing velocity task and incrementally modify it to suit your needs.
