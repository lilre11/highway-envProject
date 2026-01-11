# Quick Reference: Analysis & Enhanced Video Tools

## 📊 Model Performance Analysis

Analyze all your trained models to find which one performs best:

```bash
# Basic analysis (10 episodes per model)
python src/analyze_models.py

# Thorough analysis (30 episodes per model)
python src/analyze_models.py --episodes 30

# Compare two specific models
python src/analyze_models.py --compare model_midpoint model_final --episodes 20
```

**What it does:**
- Evaluates ALL model checkpoints
- Calculates mean rewards, collision rates, episode lengths
- Generates performance plots
- Identifies the best performing model
- Statistical comparison between models

**Output:**
- Console report with detailed metrics
- `models/performance_analysis.png` - visualization chart
- Shows if mid-training model truly outperforms final model

---

## 🎥 Enhanced Video Recording

Record videos with better visualization matching highway-env documentation:

### Three Visualization Styles

#### 1. Improved (Recommended)
```bash
python src/record_video_enhanced.py --style improved --episodes 5
```
- **Resolution**: 1200x300 (wider and taller)
- **Features**: Shows vehicle trajectories
- **Best for**: Detailed analysis

#### 2. Documentation Match
```bash
python src/record_video_enhanced.py --style documentation --episodes 5
```
- **Resolution**: 600x150 (standard)
- **Features**: Exact match to highway-env docs
- **Best for**: Comparison with documentation

#### 3. Panoramic
```bash
python src/record_video_enhanced.py --style panoramic --episodes 5
```
- **Resolution**: 1600x200 (ultra-wide)
- **Features**: Wide field of view
- **Best for**: Seeing overall traffic patterns

### Customization

```bash
# More episodes, custom FPS
python src/record_video_enhanced.py --style panoramic --episodes 10 --fps 60

# Quick preview
python src/record_video_enhanced.py --style improved --episodes 1 --fps 30
```

**Output:**
- `videos/highway_evolution_improved.mp4`
- `videos/highway_evolution_documentation.mp4`
- `videos/highway_evolution_panoramic.mp4`

---

## 🔍 Quick Diagnostic Workflow

### Step 1: Analyze Performance
```bash
python src/analyze_models.py --episodes 20
```
**Goal**: Find which model performs best

### Step 2: Visual Confirmation
```bash
python src/record_video_enhanced.py --style panoramic --episodes 5
```
**Goal**: See the behavior differences visually

### Step 3: Statistical Comparison (if needed)
```bash
python src/analyze_models.py --compare model_midpoint model_final --episodes 30
```
**Goal**: Confirm if difference is statistically significant

---

## 📈 Understanding the Results

### Performance Metrics

| Metric | Good | Concerning |
|--------|------|------------|
| Mean Reward | > 0.3 | < 0.1 |
| Collision Rate | < 10% | > 30% |
| Episode Length | > 30 steps | < 20 steps |

### Interpreting the Plot

```
Reward
  │
  │     ╭─╮
  │   ╭─╯ ╰─╮  ← Peak at 60k steps
  │ ╭─╯     ╰─╮
  │╭╯         ╰╮
  └──────────────→ Training Steps
    0   50k  100k
```

**If you see a peak then decline**: Use the checkpoint at the peak!

---

## 🛠️ Troubleshooting

### "No model checkpoints found"
```bash
# Train models first
python src/train.py
```

### "Module not found: scipy"
```bash
# Install scipy for statistical tests
pip install scipy
```

### Video looks too zoomed in
```bash
# Use panoramic style
python src/record_video_enhanced.py --style panoramic
```

### Need matplotlib for plots
```bash
pip install matplotlib
```

---

## 💡 Pro Tips

1. **Always analyze before deciding** - Don't assume the final model is best
2. **Check ALL checkpoints** - You have models at 10k, 20k, 30k, etc. The best might be at an unexpected step
3. **Use panoramic view** - Best for seeing traffic interactions
4. **Statistical significance matters** - Small differences might be random noise
5. **Consider ensemble** - You could use multiple checkpoints for different scenarios

---

## 📚 Full Documentation

See [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md) for:
- Detailed explanation of why mid-training can be better
- Training hyperparameter recommendations
- Scientific background on RL convergence
- Complete troubleshooting guide
