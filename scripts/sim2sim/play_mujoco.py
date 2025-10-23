# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.


import sys
import numpy as np
import torch

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg


# Check for --debug flag before Cfg.from_arguments() processes sys.argv
debug_mode = "--debug" in sys.argv
if debug_mode:
    sys.argv.remove("--debug")  # Remove so Cfg.from_arguments() doesn't see it

# Load configuration (Cfg.from_arguments() will parse --config)
cfg = Cfg.from_arguments()

if not cfg:
    raise ValueError("Failed to load config.")


# Main execution block
def main():
    """Main execution function for the MuJoCo simulation environment.

    Usage:
        python play_mujoco.py --config path/to/config.yaml [--debug]

    Arguments:
        --config: Path to policy configuration YAML file (required)
        --debug: Enable debug logging for gamepad and simulation state (optional)
    """
    # Initialize environment with optional debug mode
    robot = MujocoSimulator(cfg, debug=debug_mode)
    obs = robot.reset()

    # Initialize and start policy controller
    controller = RlController(cfg)
    controller.load_policy()

    # Default actions for fallback
    default_actions = np.array(cfg.default_joint_positions, dtype=np.float32)[robot.cfg.action_indices]

    # Main control loop
    while True:
        # Send observations and receive actions
        actions = controller.update(obs.numpy())

        # Use default actions if no actions received
        if actions is None:
            actions = default_actions

        # Execute step
        actions = torch.tensor(actions)
        obs = robot.step(actions)


if __name__ == "__main__":
    main()
