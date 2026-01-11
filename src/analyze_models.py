"""
Model Performance Analysis Script
Compares all checkpoints to identify best performing model and analyze training progression.

Usage:
    python src/analyze_models.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV_NAME, ENV_CONFIG, MODELS_DIR


def evaluate_model(model_path: Path, env: gym.Env, n_episodes: int = 10) -> Dict:
    """
    Evaluate a model's performance.
    
    Args:
        model_path: Path to model checkpoint
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with performance metrics
    """
    if not model_path.exists():
        return None
    
    model = PPO.load(model_path)
    
    rewards = []
    episode_lengths = []
    collisions = 0
    speeds = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        truncated = False
        episode_speeds = []
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Track speed (assuming obs contains velocity info)
            if hasattr(env.unwrapped, 'vehicle'):
                episode_speeds.append(env.unwrapped.vehicle.speed)
            
            # Check for collision
            if done and not truncated:
                collisions += 1
        
        rewards.append(total_reward)
        episode_lengths.append(steps)
        if episode_speeds:
            speeds.append(np.mean(episode_speeds))
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(episode_lengths),
        'collision_rate': collisions / n_episodes,
        'mean_speed': np.mean(speeds) if speeds else 0,
        'rewards': rewards
    }


def analyze_all_checkpoints(n_episodes: int = 10) -> None:
    """
    Analyze all model checkpoints and generate comparison report.
    
    Args:
        n_episodes: Number of episodes per model evaluation
    """
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Create evaluation environment
    env_config = ENV_CONFIG.copy()
    env = gym.make(ENV_NAME, config=env_config, render_mode=None)
    
    # Get all model files
    model_files = sorted(MODELS_DIR.glob("*.zip"))
    
    if not model_files:
        print("\n[Error] No model checkpoints found!")
        print(f"[Error] Please train models first: python src/train.py")
        return
    
    print(f"\n[Evaluating] Found {len(model_files)} checkpoints")
    print(f"[Evaluating] Episodes per model: {n_episodes}")
    print("-" * 80)
    
    results = {}
    
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\n[Progress] Evaluating: {model_name}")
        
        metrics = evaluate_model(model_file, env, n_episodes)
        if metrics:
            results[model_name] = metrics
            print(f"  Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
            print(f"  Mean Length: {metrics['mean_length']:.1f} steps")
            print(f"  Collision Rate: {metrics['collision_rate']*100:.1f}%")
            if metrics['mean_speed'] > 0:
                print(f"  Mean Speed: {metrics['mean_speed']:.2f} m/s")
    
    env.close()
    
    if not results:
        print("\n[Error] No models could be evaluated!")
        return
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['mean_reward'])
    print("\n" + "=" * 80)
    print("BEST PERFORMING MODEL")
    print("=" * 80)
    print(f"\nModel: {best_model[0]}")
    print(f"Mean Reward: {best_model[1]['mean_reward']:.3f} ± {best_model[1]['std_reward']:.3f}")
    print(f"Mean Length: {best_model[1]['mean_length']:.1f} steps")
    print(f"Collision Rate: {best_model[1]['collision_rate']*100:.1f}%")
    if best_model[1]['mean_speed'] > 0:
        print(f"Mean Speed: {best_model[1]['mean_speed']:.2f} m/s")
    
    # Plot results
    plot_results(results)
    
    print("\n[Complete] Analysis finished!")
    print("=" * 80)


def plot_results(results: Dict) -> None:
    """
    Create visualization plots of model performance.
    
    Args:
        results: Dictionary of model results
    """
    # Extract checkpoint numbers for ordering
    def get_step_number(name: str) -> int:
        if 'untrained' in name:
            return 0
        elif 'midpoint' in name:
            return 50000
        elif 'final' in name:
            return 100000
        else:
            # Extract from checkpoint_XXXXX_steps
            try:
                return int(name.split('_')[1])
            except:
                return 999999
    
    # Sort results by training steps
    sorted_results = sorted(results.items(), key=lambda x: get_step_number(x[0]))
    
    names = [name for name, _ in sorted_results]
    steps = [get_step_number(name) for name, _ in sorted_results]
    mean_rewards = [metrics['mean_reward'] for _, metrics in sorted_results]
    std_rewards = [metrics['std_reward'] for _, metrics in sorted_results]
    collision_rates = [metrics['collision_rate'] * 100 for _, metrics in sorted_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Reward progression
    ax1 = axes[0]
    ax1.errorbar(steps, mean_rewards, yerr=std_rewards, 
                 marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance vs Training Steps', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Highlight best model
    best_idx = mean_rewards.index(max(mean_rewards))
    ax1.scatter([steps[best_idx]], [mean_rewards[best_idx]], 
                color='red', s=200, marker='*', zorder=5, label='Best Model')
    ax1.legend(fontsize=10)
    
    # Plot 2: Collision rate
    ax2 = axes[1]
    ax2.plot(steps, collision_rates, marker='s', linewidth=2, 
             markersize=8, color='orange')
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Collision Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Collision Rate vs Training Steps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = PROJECT_ROOT / 'models' / 'performance_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[Plot] Performance visualization saved: {output_path}")
    
    # Also show plot
    plt.show()


def compare_specific_models(model1: str, model2: str, n_episodes: int = 20) -> None:
    """
    Detailed comparison between two specific models.
    
    Args:
        model1: First model name (without .zip)
        model2: Second model name (without .zip)
        n_episodes: Number of episodes for comparison
    """
    print("\n" + "=" * 80)
    print(f"DETAILED COMPARISON: {model1} vs {model2}")
    print("=" * 80)
    
    env_config = ENV_CONFIG.copy()
    env = gym.make(ENV_NAME, config=env_config, render_mode=None)
    
    model1_path = MODELS_DIR / f"{model1}.zip"
    model2_path = MODELS_DIR / f"{model2}.zip"
    
    results1 = evaluate_model(model1_path, env, n_episodes)
    results2 = evaluate_model(model2_path, env, n_episodes)
    
    env.close()
    
    if not results1 or not results2:
        print("[Error] One or both models not found!")
        return
    
    print(f"\n{model1}:")
    print(f"  Mean Reward: {results1['mean_reward']:.3f} ± {results1['std_reward']:.3f}")
    print(f"  Collision Rate: {results1['collision_rate']*100:.1f}%")
    
    print(f"\n{model2}:")
    print(f"  Mean Reward: {results2['mean_reward']:.3f} ± {results2['std_reward']:.3f}")
    print(f"  Collision Rate: {results2['collision_rate']*100:.1f}%")
    
    # Statistical comparison
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(results1['rewards'], results2['rewards'])
    
    print(f"\nStatistical Test (t-test):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        better_model = model1 if results1['mean_reward'] > results2['mean_reward'] else model2
        print(f"  Result: {better_model} is significantly better (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    
    print("=" * 80)


def main():
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trained models')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes per model (default: 10)')
    parser.add_argument('--compare', nargs=2, metavar=('MODEL1', 'MODEL2'),
                       help='Compare two specific models')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_specific_models(args.compare[0], args.compare[1], args.episodes)
    else:
        analyze_all_checkpoints(args.episodes)


if __name__ == "__main__":
    main()
