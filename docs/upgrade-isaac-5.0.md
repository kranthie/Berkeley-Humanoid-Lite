# Upgrade to Isaac Sim 5.0 + Isaac Lab 2.2

## Summary

Upgraded from Isaac Sim 4.5.0/Isaac Lab 2.1.0 to Isaac Sim 5.0.0/Isaac Lab 2.2.0 to enable support for NVIDIA Blackwell GPU architecture (sm_120).

## Reason

PyTorch 2.5.1 (original version) only supports CUDA compute capabilities up to sm_90. NVIDIA RTX PRO 6000 Blackwell GPU requires sm_120 support, which needs PyTorch 2.7+ with CUDA 12.8+.

## Main Changes

### Dependency Upgrades (pyproject.toml)

| Component | Old | New |
|-----------|-----|-----|
| Python | 3.10 | 3.11 |
| PyTorch | 2.5.1 (CUDA 12.1) | 2.7.0 (CUDA 12.8) |
| torchvision | 0.20.1 | 0.22.0 |
| Isaac Sim | 4.5.0 | 5.0.0 |
| Isaac Lab | 2.1.0 | 2.2.0 |
| ONNX Runtime | onnxruntime | onnxruntime-gpu>=1.22.0 |

### Code Changes

**File**: `source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/__init__.py`

Added path setup code to handle Isaac Lab 2.2's new module structure where `isaaclab_tasks` and `isaaclab_rl` are bundled inside `isaaclab/source/` instead of being directly importable.

## Migration

```bash
# Remove old environment
rm -rf .venv

# Reinstall with new dependencies
uv sync
```

## Notes

- Isaac Lab 2.2 requires AppLauncher initialization before imports work properly
- ONNX Runtime GPU provides GPU-accelerated inference (CPU fallback included)
- For details, see the git commit that made these changes
