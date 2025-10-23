"""Package containing task implementations for various robotic environments."""

import os
import sys

# Add Isaac Lab source extensions to Python path for Isaac Lab 2.2
try:
    import isaaclab
    isaaclab_path = os.path.dirname(os.path.abspath(isaaclab.__file__))
    source_path = os.path.join(isaaclab_path, "source")

    # Add isaaclab core to path (for isaaclab.envs, etc.)
    isaaclab_core_path = os.path.join(source_path, "isaaclab")
    if os.path.exists(isaaclab_core_path) and isaaclab_core_path not in sys.path:
        sys.path.insert(0, isaaclab_core_path)

    # Add isaaclab_tasks and isaaclab_rl to path
    for extension in ["isaaclab_tasks", "isaaclab_rl", "isaaclab_assets", "isaaclab_mimic"]:
        ext_path = os.path.join(source_path, extension)
        if os.path.exists(ext_path) and ext_path not in sys.path:
            sys.path.insert(0, ext_path)
except ImportError:
    pass

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
