# Training Guide

Quick reference for training policies for Berkeley Humanoid Lite.

## Available Tasks

1. **Velocity-Berkeley-Humanoid-Lite-v0** - Full humanoid (22 DOF)
   - Controls torso, arms, and legs
   - Learns to walk at commanded velocities

2. **Velocity-Berkeley-Humanoid-Lite-Biped-v0** - Legs only (12 DOF)
   - Controls leg joints only
   - Simpler, faster to train

## Basic Training

```bash
# Train full humanoid
uv run ./scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-v0 \
    --num_envs 4096 \
    --headless

# Train biped (legs only)
uv run ./scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --num_envs 4096 \
    --headless
```

### Common Options

- `--num_envs <N>` - Number of parallel environments (default: 4096)
- `--headless` - Run without GUI (better performance)
- `--video` - Record training videos
- `--max_iterations <N>` - Override default iteration count (default: 6000)

## Monitoring Training

### Launch TensorBoard

```bash
tensorboard --logdir logs/rsl_rl
```

Then open http://localhost:6006 to view:
- Episode rewards and lengths
- Learning curves (loss, entropy)
- Training statistics
- Recorded videos (if `--video` was used)

### Console Output

Training prints progress every few iterations:
- Current iteration
- Mean episode reward
- Episode length
- Training time

## After Training

### Export Policy

```bash
uv run ./scripts/rsl_rl/play.py \
    --task Velocity-Berkeley-Humanoid-Lite-v0 \
    --num_envs 1
```

This automatically:
- Loads the latest checkpoint
- Exports to ONNX format: `logs/rsl_rl/<experiment>/exported/policy.onnx`
- Exports to TorchScript: `logs/rsl_rl/<experiment>/exported/policy.pt`
- Saves deployment config: `configs/policy_latest.yaml`

### Validate in MuJoCo

```bash
# Test the exported policy in lightweight MuJoCo simulator
uv run ./scripts/sim2sim/play_mujoco.py --config configs/policy_latest.yaml
```

## Training Output

Logs are saved to:
```
logs/rsl_rl/<task_name>/<timestamp>/
├── checkpoints/           # Model checkpoints
├── exported/              # ONNX and TorchScript policies
├── events.out.tfevents.*  # TensorBoard logs
└── videos/                # Recorded videos (if --video used)
```

## Performance Tips

### GPU Memory

If you run out of GPU memory:
- Reduce `--num_envs` (try 2048 or 1024)
- Use `--headless` to save VRAM
- Close other GPU applications

### Training Speed

- More environments = faster learning (diminishing returns above 4096)
- Headless mode is ~20-30% faster than GUI mode
- Default 6000 iterations takes ~2 hours on modern GPU

### Quality

- Full humanoid (22 DOF) takes longer to learn than biped (12 DOF)
- Monitor reward curves in TensorBoard
- If rewards plateau early, check reward function weights
- Training is stochastic - run multiple seeds if needed

## Troubleshooting

**Training crashes immediately**
- Check GPU supports CUDA 12.8 (Blackwell needs PyTorch 2.7+)
- Verify environment with `uv sync`
- Try reducing `--num_envs`

**Poor performance/low rewards**
- Check TensorBoard for learning curves
- Verify task config in `source/berkeley_humanoid_lite/tasks/`
- Try adjusting reward weights (see policies-guide.md)

**Out of memory**
- Reduce `--num_envs`
- Use `--headless`
- Close other applications

**Can't find checkpoint**
- Run `play.py` - it finds the latest checkpoint automatically
- Or specify with `--load_run <run_name>`
