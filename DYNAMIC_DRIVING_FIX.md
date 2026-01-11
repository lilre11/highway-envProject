# Fix: Cars Going Straight (Boring Behavior)

## Problem
Your trained agent just drives straight without changing lanes or showing dynamic behavior.

## Root Cause
The reward configuration encouraged **safe but boring** driving:
- ❌ `lane_change_reward: 0.0` - No incentive to change lanes
- ❌ `right_lane_reward: 0.1` - Encouraged staying in one lane
- ❌ Heavy collision penalty made agent risk-averse
- ❌ Low exploration (`ent_coef: 0.01`)

**Result**: Agent learned "just go straight = safe rewards, no risk"

## Changes Made

### ✅ Reward Configuration
```python
"right_lane_reward": 0.0,          # Removed (was 0.1)
"lane_change_reward": 0.1,         # Added (was 0.0)
"on_road_reward": 0.1,             # Added for safe maneuvering
"vehicles_density": 1.5,           # Increased traffic (was implicit 1.0)
```

### ✅ Training Configuration
```python
"ent_coef": 0.02,                  # Increased exploration (was 0.01)
```

### What These Changes Do

1. **Lane Change Reward (0.1)**
   - Now the agent gets rewarded for changing lanes
   - Encourages overtaking and dynamic behavior
   
2. **Removed Right Lane Bias (0.0)**
   - Agent was "anchored" to right lanes
   - Now free to use all lanes strategically
   
3. **Increased Traffic Density (1.5)**
   - More obstacles require more maneuvering
   - Can't just cruise in empty lane
   
4. **Higher Exploration (ent_coef: 0.02)**
   - Agent tries more actions during training
   - Discovers lane-changing strategies

## Expected New Behavior

### Before (Current Models)
```
Lane 1: |                |
Lane 2: |    🚗→→→→→→   |  (Agent just goes straight)
Lane 3: |                |
Lane 4: |                |
```

### After (Retrained Models)
```
Lane 1: |      🚗        |
Lane 2: |    ↗  ↘        |  (Agent maneuvers around traffic)
Lane 3: |  ↗      ↘      |
Lane 4: |                |
```

## How to Retrain

```bash
# Step 1: Backup old models (optional)
mkdir models_old
move models\*.zip models_old\

# Step 2: Retrain with new configuration
python src/train.py

# Step 3: Record new videos
python src/record_video_enhanced.py --style panoramic --episodes 5

# Step 4: Compare performance
python src/analyze_models.py --episodes 20
```

## What to Expect

### Training Metrics
- **Initial episodes**: More crashes (agent experimenting with lane changes)
- **Mid-training**: Finding balance between speed and safety
- **Final model**: Dynamic lane changes + safe driving

### Performance Comparison

| Metric | Old (Straight) | New (Dynamic) | Note |
|--------|----------------|---------------|------|
| Mean Reward | ~31 | ~28-33 | Might be slightly lower initially |
| Collision Rate | 0% | 5-10% | Small increase acceptable for dynamic behavior |
| Lane Changes | ~0 per episode | 3-5 per episode | **Key improvement!** |
| Visual Interest | ⭐ | ⭐⭐⭐⭐⭐ | Much more interesting to watch |

## If Results Are Not Dynamic Enough

### Option 1: Increase Lane Change Reward Further
```python
"lane_change_reward": 0.2,  # Even more incentive
```

### Option 2: Add Penalty for Staying in Same Lane
```python
"lane_change_reward": 0.15,
"same_lane_penalty": -0.05,  # Penalize boring driving
```

### Option 3: Increase Traffic Even More
```python
"vehicles_count": 70,      # More cars (was 50)
"vehicles_density": 2.0,   # Denser traffic (was 1.5)
```

### Option 4: Use Different Environment
```python
ENV_NAME: str = "highway-v0"  # Instead of "highway-fast-v0"
# highway-v0 has more challenging traffic patterns
```

## Training Tips

1. **Watch TensorBoard**: Monitor if lane changes are increasing
   ```bash
   tensorboard --logdir=logs
   ```

2. **Early checkpoints matter**: Check 30k-40k step models first
   - They might show lane changes before over-optimization

3. **Increase total timesteps**: Dynamic behavior needs more learning
   ```python
   "total_timesteps": 150_000  # Give it more time
   ```

4. **Record during training**: See behavior evolution
   ```bash
   # After 50k steps
   python src/record_video_enhanced.py --style panoramic --episodes 3
   ```

## Rollback Instructions

If new behavior is worse:
```bash
# Restore old models
move models_old\*.zip models\

# Revert config.py
git checkout config.py
```

## Scientific Explanation

This is a classic **reward shaping** problem in RL:
- **Sparse rewards**: Agent finds local optimum (going straight)
- **Dense rewards**: Agent explores full action space (lane changes)

By adding `lane_change_reward`, we're implementing **reward shaping** to guide the agent toward more interesting behaviors while maintaining safety.

## Next Steps

1. ✅ Retrain with new configuration
2. ✅ Compare old vs new behavior visually
3. ✅ Measure lane change frequency
4. 🎯 Fine-tune if needed
