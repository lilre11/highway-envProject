"""
Generate training curves from TensorBoard logs for README visualization.

Usage:
    python src/plot_training.py
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_tensorboard_data(log_dir: Path, tag: str):
    """Load specific metric from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except KeyError:
        return [], []


def plot_training_curves():
    """Generate training performance plots."""
    logs_dir = PROJECT_ROOT / "logs"
    
    # Find the latest PPO run
    ppo_runs = sorted(logs_dir.glob("PPO_*"))
    if not ppo_runs:
        print("No training logs found. Please run training first.")
        return
    
    latest_run = ppo_runs[-1]
    print(f"Loading logs from: {latest_run}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Performance Curves', fontsize=16, fontweight='bold')
    
    # Metrics to plot
    metrics = [
        ('rollout/ep_rew_mean', 'Episode Reward', 'Mean Reward'),
        ('rollout/ep_len_mean', 'Episode Length', 'Mean Steps'),
        ('train/learning_rate', 'Learning Rate', 'Learning Rate'),
        ('train/entropy_loss', 'Entropy Loss', 'Entropy')
    ]
    
    for idx, (tag, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        steps, values = load_tensorboard_data(latest_run, tag)
        
        if steps and values:
            ax.plot(steps, values, linewidth=2, color='#2E86AB')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(steps) > 1:
                z = np.polyfit(steps, values, 3)
                p = np.poly1d(z)
                ax.plot(steps, p(steps), "--", color='#FF6B35', alpha=0.7, 
                       label='Trend', linewidth=1.5)
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for {tag}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path = PROJECT_ROOT / "models" / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Training curves saved: {output_path}")
    
    # Also show the plot
    plt.show()


if __name__ == "__main__":
    plot_training_curves()
