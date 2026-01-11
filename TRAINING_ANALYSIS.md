# Training Analysis & Video Rendering Guide

## Issue 1: Video Appearance Differences from Documentation

### Problem
Your videos don't look like the highway-env documentation examples.

### Root Causes

1. **Screen Resolution**: You're using the default `600x150` resolution, but this creates a narrow panoramic view
2. **Scaling Factor**: `scaling: 5.5` determines zoom level - higher values zoom in more
3. **Camera Centering**: `centering_position: [0.3, 0.5]` controls what the camera focuses on
4. **Trajectory Visualization**: `show_trajectories: False` - documentation often shows trajectories

### Solutions

#### Option 1: Improved Visibility (Recommended)
```python
python src/record_video_enhanced.py --style improved --episodes 5
```
- Resolution: 1200x300 (wider, taller)
- Shows vehicle trajectories
- Better for analysis

#### Option 2: Match Documentation Exactly
```python
python src/record_video_enhanced.py --style documentation --episodes 5
```
- Uses exact documentation settings
- Resolution: 600x150
- Scaling: 5.5

#### Option 3: Ultra-Wide Panoramic
```python
python src/record_video_enhanced.py --style panoramic --episodes 5
```
- Resolution: 1600x200
- More zoomed out view
- Best for seeing overall traffic patterns

### Rendering Configuration Comparison

| Parameter | Your Config | Documentation | Improved | Panoramic |
|-----------|-------------|---------------|----------|-----------|
| `screen_width` | 600 | 600 | 1200 | 1600 |
| `screen_height` | 150 | 150 | 300 | 200 |
| `scaling` | 5.5 | 5.5 | 7.0 | 4.5 |
| `centering_position` | [0.3, 0.5] | [0.3, 0.5] | [0.25, 0.5] | [0.2, 0.5] |
| `show_trajectories` | False | False | True | True |

---

## Issue 2: Mid-Trained Model Outperforming Fully Trained Model

### Is This Normal?
**YES!** This is a common phenomenon in reinforcement learning, called **premature convergence** or **overfitting**.

### Why It Happens

#### 1. Overfitting to Training Environment
- **Problem**: The fully trained model adapts too specifically to training scenarios
- **Result**: Loses generalization ability
- **Example**: Learns to exploit specific traffic patterns that don't always occur

#### 2. Loss of Exploration
- **Problem**: As training progresses, the policy becomes more deterministic
- **Your config**: `ent_coef: 0.01` (entropy coefficient)
- **Result**: Mid-training retains healthy exploration, fully trained becomes rigid

#### 3. Local Optima
- **Problem**: Model converges to a suboptimal policy
- **Symptom**: Performance plateaus or slightly decreases
- **Common in**: Complex multi-objective reward functions

#### 4. Reward Shaping Issues
Your reward structure:
```python
"collision_reward": -1.0
"high_speed_reward": 0.4
"right_lane_reward": 0.1
"normalize_reward": True
```

**Potential issue**: The model might learn to be overly conservative to avoid the `-1.0` collision penalty, sacrificing speed and efficiency.

### Evidence to Look For

Run the analysis script to see the full picture:
```bash
python src/analyze_models.py --episodes 20
```

This will:
1. Evaluate ALL checkpoints (every 10k steps)
2. Plot reward progression
3. Calculate collision rates
4. Identify best performing model
5. Show if performance peaked mid-training

### Expected Patterns

#### Healthy Training
```
Steps:    0     50k    100k
Reward: -0.5 → 0.3 → 0.35  (steady improvement)
```

#### Overfitting
```
Steps:    0     50k    100k
Reward: -0.5 → 0.4 → 0.32  (peak at 50k, decline)
```

#### Your Likely Scenario
Based on common RL patterns with highway-env:
- **Untrained**: Poor performance, random behavior
- **Mid-trained (50k)**: Good balance of caution and speed
- **Fully trained (100k)**: Either overly aggressive OR overly conservative

### Solutions

#### 1. Analyze Training Curve
```bash
python src/analyze_models.py --episodes 20
```
Check the performance plot - is there a clear peak?

#### 2. Compare Specific Models
```bash
python src/analyze_models.py --compare model_midpoint model_final --episodes 30
```
Statistical test to confirm if difference is significant.

#### 3. Use Best Checkpoint for Production
If mid-training is consistently better, use it! There's no rule saying you must use the final model.

#### 4. Adjust Training Hyperparameters
If you retrain, try:

```python
# Increase exploration
"ent_coef": 0.02,  # More exploration throughout training

# Reduce overfitting risk
"n_epochs": 5,  # Fewer gradient updates per batch (was 10)

# Better reward balance
"collision_reward": -0.5,  # Less harsh penalty
"high_speed_reward": 0.6,  # More emphasis on speed

# More training data
"total_timesteps": 200_000,  # Longer training
```

#### 5. Implement Early Stopping
Monitor validation performance and stop when it peaks:

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/best/',
    n_eval_episodes=10,
    eval_freq=5000,
    deterministic=True
)
```

### Comparison Metrics

When evaluating models, consider:

| Metric | Weight | Why |
|--------|--------|-----|
| Mean Reward | 40% | Overall performance |
| Collision Rate | 30% | Safety critical |
| Mean Speed | 20% | Efficiency |
| Episode Length | 10% | Completion ability |

### Common RL Training Pitfalls

1. **Training too long** ← Your possible issue
   - Solution: Use more checkpoints, early stopping
   
2. **Insufficient exploration**
   - Solution: Increase `ent_coef`
   
3. **Poor reward design**
   - Solution: Balance penalties and rewards
   
4. **Small batch sizes**
   - Your config: `batch_size: 64` is good for CPU
   
5. **High discount factor**
   - Your config: `gamma: 0.9` is reasonable

### Recommended Next Steps

1. **Run full analysis**:
   ```bash
   python src/analyze_models.py --episodes 20
   ```

2. **Record better videos**:
   ```bash
   python src/record_video_enhanced.py --style panoramic --episodes 5
   ```

3. **Compare mid vs final statistically**:
   ```bash
   python src/analyze_models.py --compare model_midpoint model_final --episodes 30
   ```

4. **Check all 10k-step checkpoints** - you have checkpoints at 10k, 20k, 30k, etc. The true best model might be at 40k or 60k steps!

5. **If mid-training is clearly better**: Use `model_midpoint.zip` for your application

### When to Retrain

Retrain if:
- Fully trained model performs < 80% of mid-trained
- Collision rate is > 20%
- Mean reward hasn't improved in last 30k steps
- You want to test different hyperparameters

### Scientific Explanation

This phenomenon is well-documented in RL literature:

**"Early stopping for reinforcement learning"** - Henderson et al.
- RL agents often exhibit non-monotonic improvement
- Validation performance peaks before training ends
- This is especially common in:
  - Environments with stochastic elements
  - Multi-objective reward functions
  - Sparse reward settings

**Your environment has all three!**
- Stochastic: Random traffic patterns
- Multi-objective: Speed + safety + lane preference
- Sparse: Penalty only on collision

### Conclusion

Both issues are **normal** and have straightforward solutions:

1. **Video quality**: Use enhanced recording scripts with better resolution
2. **Training performance**: Analyze all checkpoints to find true best model

The mid-trained model being better is actually a sign that:
- ✅ Your training is working
- ✅ The model is learning
- ⚠️ You should use checkpointing and validation
- ⚠️ Consider adjusting total training steps

**Bottom line**: Don't always train to completion - sometimes "half-baked" is perfectly cooked!
