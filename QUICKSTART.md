# 🚦 Highway-Env RL - Quick Start Guide

## Installation (5 minutes)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup
```bash
python config.py
```

You should see configuration details printed without errors.

---

## Training Workflow

### Phase 1: Train the Agent (25-30 min on CPU)
```bash
python src/train.py
```

**What happens:**
- Creates `highway-fast-v0` environment
- Saves untrained model immediately
- Trains PPO for 100k timesteps
- Saves checkpoints every 10k steps
- Saves midpoint model at 50k steps
- Saves final trained model at 100k steps
- Evaluates performance on 5 episodes

**Expected console output:**
```
================================================================================
HIGHWAY-ENV REINFORCEMENT LEARNING - PPO TRAINING
================================================================================

[Environment] Created: highway-fast-v0
[Environment] Observation Space: Box(...)
[Environment] Action Space: Discrete(5)

[Checkpoint] Untrained model saved: models/model_untrained.zip

[Model] PPO initialized with hyperparameters:
  - Learning Rate: 0.0005
  - Batch Size: 64
  - Gamma: 0.9
  - Network Architecture: [256, 256]

[Training] Starting training for 100,000 timesteps
[Training] Callbacks enabled: EvolutionCheckpoint, CheckpointCallback
[Training] TensorBoard logs: tensorboard --logdir logs/

[Progress bar showing training...]
```

**Checkpoints saved:**
- `models/model_untrained.zip` (baseline)
- `models/model_midpoint.zip` (50k steps)
- `models/model_final.zip` (100k steps)
- `models/checkpoint_10000_steps.zip` (every 10k)

---

### Phase 2: Monitor Training (Optional)

Open a new terminal and run:
```bash
tensorboard --logdir logs/
```

Then open your browser to: http://localhost:6006

**You'll see:**
- Reward curves over time
- Policy loss, value loss
- Entropy (exploration) decay
- Learning rate schedule

---

### Phase 3: Generate Evolution Video (2-3 min)
```bash
python src/record_video.py
```

**What happens:**
- Loads untrained, midpoint, and final models
- Records 3 episodes per checkpoint
- Adds text overlays (model state, reward)
- Saves video to `videos/highway_evolution_evolution.mp4`

**Console output:**
```
================================================================================
RECORDING MODEL EVOLUTION VIDEO
================================================================================

[Load] Loading model: model_untrained.zip
[Recording] Untrained Agent
----------------------------------------
  Episode 1/3 | Reward: 0.15 | Steps: 127 | Frames: 127
  Episode 2/3 | Reward: 0.12 | Steps: 89 | Frames: 89
  Episode 3/3 | Reward: 0.18 | Steps: 156 | Frames: 156

[Load] Loading model: model_midpoint.zip
[Recording] Mid-Training (50%)
----------------------------------------
  Episode 1/3 | Reward: 0.28 | Steps: 285 | Frames: 285
  ...

[Saving] Generating video: videos/highway_evolution_evolution.mp4
[Success] Video saved successfully
```

---

### Phase 4: Compare Models (Optional)
```bash
python src/utils.py
```

Evaluates all three checkpoints and prints comparison table:
```
================================================================================
MODEL COMPARISON
================================================================================

[Evaluating] model_untrained
  Mean Reward: 0.153 ± 0.042

[Evaluating] model_midpoint
  Mean Reward: 0.287 ± 0.065

[Evaluating] model_final
  Mean Reward: 0.391 ± 0.052

================================================================================
SUMMARY
================================================================================
Model                          Mean Reward     Std Reward     
--------------------------------------------------------------------------------
model_untrained                0.153           0.042          
model_midpoint                 0.287           0.065          
model_final                    0.391           0.052          
================================================================================
```

---

## File Overview

### Core Files (Must Read)
- **`config.py`**: All hyperparameters in one place
- **`src/train.py`**: Complete training pipeline
- **`src/record_video.py`**: Video generation
- **`README.md`**: Full methodology & documentation

### Generated Files
- **`models/*.zip`**: Model checkpoints
- **`videos/*.mp4`**: Evolution videos
- **`logs/`**: TensorBoard logs

---

## Customization Tips

### To Train Longer (Better Results):
Edit `config.py`:
```python
TRAINING_CONFIG: Dict[str, Any] = {
    "total_timesteps": 200_000,  # Changed from 100k
    ...
}
```

### To Change Environment Difficulty:
Edit `config.py`:
```python
ENV_CONFIG: Dict[str, Any] = {
    "vehicles_count": 80,  # More traffic (harder)
    "lanes_count": 5,      # More lanes
    "duration": 60,        # Longer episodes
    ...
}
```

### To Use GPU (Much Faster):
No code changes needed! PyTorch automatically detects CUDA.
Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'highway_env'`
**Solution:**
```bash
pip install highway-env
```

### Issue: Training is too slow on CPU
**Solutions:**
1. Reduce `total_timesteps` to 50k for testing
2. Reduce `n_steps` to 1024 in `config.py`
3. Use smaller network: `"net_arch": [128, 128]`

### Issue: Video recording fails
**Solution:**
```bash
pip install imageio-ffmpeg
```

### Issue: `Model not found` error when recording
**Solution:** Run training first:
```bash
python src/train.py
```

---

## What to Submit for Grading

### 1. Source Code ✅
- `config.py`
- `src/train.py`
- `src/record_video.py`
- `README.md`

### 2. Results ✅
- Training logs (TensorBoard screenshots)
- Evolution video (`videos/*.mp4`)
- Final model checkpoint (`models/model_final.zip`)

### 3. Report Sections ✅
- **Methodology**: Already in README.md
- **Reward Function**: LaTeX formula included
- **PPO Justification**: DQN comparison included
- **Results**: Add your actual training curves and final metrics

---

## Expected Timeline

| Phase | Duration | Activity |
|-------|----------|----------|
| Setup | 5 min | Install dependencies |
| Training | 25-30 min | Train PPO agent |
| Video | 2-3 min | Generate evolution video |
| Analysis | 10 min | Review results, TensorBoard |
| **Total** | **~45 min** | Complete pipeline |

---

## Next Steps After Basic Training

1. **Experiment with Hyperparameters**: Try different learning rates, network sizes
2. **Longer Training**: Increase to 200k-500k timesteps
3. **Different Observations**: Try image-based observations (CNN)
4. **Compare Algorithms**: Implement DQN or SAC for comparison
5. **Multi-Agent**: Extend to control multiple vehicles

---

## Support

If you encounter issues:
1. Check [Highway-Env Docs](https://highway-env.farama.org/)
2. Check [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
3. Verify Python version: `python --version` (should be 3.9+)
4. Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

---

**Ready to start? Run:**
```bash
python src/train.py
```

Good luck with your capstone! 🎓🚗💨
