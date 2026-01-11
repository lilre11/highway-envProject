"""
Utility functions for the Highway-Env RL project.
"""

import sys
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import highway_env  # Required to register highway-env environments
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV_NAME, ENV_CONFIG, get_model_path


def evaluate_model_statistics(
    model: PPO,
    env: gym.Env,
    n_eval_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a model and return detailed statistics.

    Args:
        model: Trained PPO model
        env: Gymnasium environment
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy

    Returns:
        Dictionary containing evaluation statistics
    """
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )

    results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_episodes": n_eval_episodes,
    }

    return results


def compare_models(model_names: list[str], n_eval_episodes: int = 10) -> None:
    """
    Compare multiple model checkpoints and print results.

    Args:
        model_names: List of model checkpoint names
        n_eval_episodes: Number of episodes per model
    """
    env = gym.make(ENV_NAME, config=ENV_CONFIG)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    results = []

    for model_name in model_names:
        model_path = get_model_path(model_name)
        
        if not model_path.exists():
            print(f"\n[Warning] Model not found: {model_name}")
            continue

        print(f"\n[Evaluating] {model_name}")
        model = PPO.load(model_path)
        
        stats = evaluate_model_statistics(
            model, env, n_eval_episodes=n_eval_episodes
        )
        
        results.append({
            "name": model_name,
            "mean_reward": stats["mean_reward"],
            "std_reward": stats["std_reward"],
        })

        print(f"  Mean Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")

    env.close()

    # Print summary table
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Model':<30} {'Mean Reward':<15} {'Std Reward':<15}")
        print("-" * 80)
        for result in results:
            print(f"{result['name']:<30} "
                  f"{result['mean_reward']:<15.3f} "
                  f"{result['std_reward']:<15.3f}")
        print("=" * 80)


if __name__ == "__main__":
    # Compare evolution checkpoints
    from config import CHECKPOINT_CONFIG
    
    models_to_compare = [
        CHECKPOINT_CONFIG["untrained_name"],
        CHECKPOINT_CONFIG["midpoint_name"],
        CHECKPOINT_CONFIG["final_name"],
    ]
    
    compare_models(models_to_compare, n_eval_episodes=10)
