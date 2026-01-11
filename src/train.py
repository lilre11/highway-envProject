"""
Training script for Highway-Env using DQN from Stable-Baselines3.
Includes checkpoint callback mechanism for generating evolution videos.

Usage:
    python src/train.py
"""

import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ENV_NAME,
    ENV_CONFIG,
    TRAINING_CONFIG,
    CHECKPOINT_CONFIG,
    get_model_path,
)


class EvolutionCheckpointCallback(BaseCallback):
    """
    Custom callback to save specific checkpoints for evolution video.
    Saves untrained, midpoint, and final model states.
    """

    def __init__(
        self,
        total_timesteps: int,
        checkpoint_dir: Path,
        verbose: int = 0
    ):
        """
        Initialize the evolution checkpoint callback.

        Args:
            total_timesteps: Total number of training timesteps
            checkpoint_dir: Directory to save checkpoints
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.checkpoint_dir = checkpoint_dir
        self.midpoint = total_timesteps // 2
        self.saved_midpoint = False

    def _on_step(self) -> bool:
        """
        Called at each training step. Saves midpoint checkpoint.

        Returns:
            True to continue training
        """
        # Save midpoint model
        if not self.saved_midpoint and self.num_timesteps >= self.midpoint:
            midpoint_path = get_model_path(
                CHECKPOINT_CONFIG["midpoint_name"]
            )
            self.model.save(midpoint_path)
            if self.verbose > 0:
                print(f"\n[Checkpoint] Midpoint model saved at step {self.num_timesteps}")
                print(f"[Checkpoint] Path: {midpoint_path}")
            self.saved_midpoint = True

        return True


def create_environment() -> gym.Env:
    """
    Create and configure the Highway-Env environment.

    Returns:
        Configured Gymnasium environment
    """
    env = gym.make(ENV_NAME, config=ENV_CONFIG, render_mode="rgb_array")
    print(f"\n[Environment] Created: {ENV_NAME}")
    print(f"[Environment] Observation Space: {env.observation_space}")
    print(f"[Environment] Action Space: {env.action_space}")
    return env


def save_untrained_model(env: gym.Env) -> None:
    """
    Save an untrained model for comparison in evolution video.

    Args:
        env: Gymnasium environment
    """
    untrained_model = DQN(
        "MlpPolicy",
        env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        buffer_size=TRAINING_CONFIG.get("buffer_size", 50000),
        learning_starts=TRAINING_CONFIG.get("learning_starts", 1000),
        batch_size=TRAINING_CONFIG["batch_size"],
        gamma=TRAINING_CONFIG["gamma"],
        train_freq=TRAINING_CONFIG.get("train_freq", 4),
        gradient_steps=TRAINING_CONFIG.get("gradient_steps", 1),
        target_update_interval=TRAINING_CONFIG.get("target_update_interval", 1000),
        exploration_fraction=TRAINING_CONFIG.get("exploration_fraction", 0.1),
        exploration_initial_eps=TRAINING_CONFIG.get("exploration_initial_eps", 1.0),
        exploration_final_eps=TRAINING_CONFIG.get("exploration_final_eps", 0.05),
        policy_kwargs=TRAINING_CONFIG["policy_kwargs"],
        verbose=0,
    )
    untrained_path = get_model_path(CHECKPOINT_CONFIG["untrained_name"])
    untrained_model.save(untrained_path)
    print(f"\n[Checkpoint] Untrained model saved: {untrained_path}")


def train_model(env: gym.Env) -> DQN:
    """
    Train the DQN agent on the Highway environment.

    Args:
        env: Gymnasium environment

    Returns:
        Trained DQN model
    """
    print("\n" + "=" * 80)
    print("TRAINING DQN AGENT")
    print("=" * 80)

    # Initialize DQN model
    # Try to use tensorboard, but make it optional
    tensorboard_log = None
    try:
        import tensorboard
        tensorboard_log = TRAINING_CONFIG["tensorboard_log"]
        print("[Info] TensorBoard logging enabled")
    except ImportError:
        print("[Warning] TensorBoard not installed. Logging disabled.")
        print("[Warning] Install with: pip install tensorboard")
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        buffer_size=TRAINING_CONFIG.get("buffer_size", 50000),
        learning_starts=TRAINING_CONFIG.get("learning_starts", 1000),
        batch_size=TRAINING_CONFIG["batch_size"],
        gamma=TRAINING_CONFIG["gamma"],
        train_freq=TRAINING_CONFIG.get("train_freq", 4),
        gradient_steps=TRAINING_CONFIG.get("gradient_steps", 1),
        target_update_interval=TRAINING_CONFIG.get("target_update_interval", 1000),
        exploration_fraction=TRAINING_CONFIG.get("exploration_fraction", 0.1),
        exploration_initial_eps=TRAINING_CONFIG.get("exploration_initial_eps", 1.0),
        exploration_final_eps=TRAINING_CONFIG.get("exploration_final_eps", 0.05),
        policy_kwargs=TRAINING_CONFIG["policy_kwargs"],
        verbose=TRAINING_CONFIG["verbose"],
        tensorboard_log=tensorboard_log,
    )

    print("\n[Model] DQN initialized with hyperparameters:")
    print(f"  - Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  - Buffer Size: {TRAINING_CONFIG.get('buffer_size', 50000)}")
    print(f"  - Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"  - Gamma: {TRAINING_CONFIG['gamma']}")
    print(f"  - Exploration: {TRAINING_CONFIG.get('exploration_initial_eps', 1.0)} → {TRAINING_CONFIG.get('exploration_final_eps', 0.05)}")
    print(f"  - Network Architecture: {TRAINING_CONFIG['policy_kwargs']['net_arch']}")

    # Setup callbacks
    evolution_callback = EvolutionCheckpointCallback(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        checkpoint_dir=CHECKPOINT_CONFIG["checkpoint_dir"],
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_CONFIG["save_freq"],
        save_path=str(CHECKPOINT_CONFIG["checkpoint_dir"]),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    print(f"\n[Training] Starting training for {TRAINING_CONFIG['total_timesteps']:,} timesteps")
    print("[Training] Callbacks enabled: EvolutionCheckpoint, CheckpointCallback")
    if tensorboard_log:
        print("[Training] TensorBoard logs: tensorboard --logdir logs/")
    
    # Check if progress bar is available
    use_progress_bar = False
    try:
        import tqdm
        import rich
        use_progress_bar = True
        print("[Training] Progress bar enabled\n")
    except ImportError:
        print("[Warning] tqdm/rich not installed. Progress bar disabled.")
        print("[Warning] Install with: pip install tqdm rich\n")

    # Train the model
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=[evolution_callback, checkpoint_callback],
        progress_bar=use_progress_bar,
    )

    print("\n[Training] Training completed!")
    return model


def save_final_model(model: DQN) -> None:
    """
    Save the final trained model.

    Args:
        model: Trained DQN model
    """
    final_path = get_model_path(CHECKPOINT_CONFIG["final_name"])
    model.save(final_path)
    print(f"\n[Checkpoint] Final model saved: {final_path}")


def evaluate_model(model: DQN, env: gym.Env, num_episodes: int = 5) -> None:
    """
    Evaluate the trained model.

    Args:
        model: Trained DQN model
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
    """
    print("\n" + "=" * 80)
    print("EVALUATING TRAINED MODEL")
    print("=" * 80)

    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)
        print(f"[Eval] Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | Steps: {steps}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_steps = sum(total_steps) / len(total_steps)

    print(f"\n[Eval] Average Reward: {avg_reward:.2f}")
    print(f"[Eval] Average Steps: {avg_steps:.2f}")
    print("=" * 80)


def main() -> None:
    """Main training pipeline."""
    print("\n" + "=" * 80)
    print("HIGHWAY-ENV REINFORCEMENT LEARNING - DQN TRAINING")
    print("=" * 80)

    # Create environment
    env = create_environment()

    # Save untrained model (for evolution video)
    save_untrained_model(env)

    # Train the model
    model = train_model(env)

    # Save final model
    save_final_model(model)

    # Evaluate the trained model
    evaluate_model(model, env, num_episodes=5)

    # Cleanup
    env.close()
    print("\n[Complete] Training pipeline finished successfully!")
    print("[Next Step] Run 'python src/record_video.py' to generate evolution video")


if __name__ == "__main__":
    main()
