"""
Configuration file for Highway-Env Autonomous Driving RL Project.
All hyperparameters and environment settings are centralized here.
Complies with PEP8 standards and uses type hinting.
"""

from pathlib import Path
from typing import Dict, Any


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT: Path = Path(__file__).parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
VIDEOS_DIR: Path = PROJECT_ROOT / "videos"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

ENV_NAME: str = "highway-fast-v0"

# Highway-Env Configuration (as per official documentation)
# Reference: https://highway-env.farama.org/environments/highway/
ENV_CONFIG: Dict[str, Any] = {
    # Observation and Action Space
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    
    # Road Configuration
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "initial_spacing": 2,
    "controlled_vehicles": 1,
    
    # Reward Function Components
    "collision_reward": -0.3,  # Reduced penalty to encourage aggressive overtaking
    "reward_speed_range": [20, 35],  # Higher range for faster driving (up to 35 m/s)
    "high_speed_reward": 1.2,  # Increased to strongly incentivize high-speed driving
    "right_lane_reward": 0.0,  # Removed to encourage left lane overtaking
    "normalize_reward": True,
    
    # Simulation Parameters
    "simulation_frequency": 15,
    "policy_frequency": 1,
    
    # Vehicle Behavior
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    
    # OVERTAKING: Make other vehicles slower so agent naturally overtakes them
    "vehicles_speed": 15,  # Other cars go 15 m/s, agent targets 20-35 m/s for more overtaking
    
    # Rendering (disabled for CPU training)
    "offscreen_rendering": False,
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
}


# ============================================================================
# TRAINING HYPERPARAMETERS (DQN - CPU Optimized)
# ============================================================================

TRAINING_CONFIG: Dict[str, Any] = {
    # Total training timesteps
    "total_timesteps": 100_000,  # Increase to 200k+ for better results
    
    # DQN Hyperparameters
    "learning_rate": 5e-4,
    "buffer_size": 50_000,  # Replay buffer size
    "learning_starts": 1_000,  # Start learning after this many steps
    "batch_size": 64,  # Mini-batch size (CPU optimized)
    "gamma": 0.99,  # Discount factor (higher for long-term rewards)
    "train_freq": 4,  # Update the model every 4 steps
    "gradient_steps": 1,  # Number of gradient steps per update
    "target_update_interval": 1_000,  # Update target network every 1000 steps
    "exploration_fraction": 0.2,  # Fraction of total timesteps for exploration
    "exploration_initial_eps": 1.0,  # Initial exploration rate
    "exploration_final_eps": 0.05,  # Final exploration rate
    
    # Neural Network Architecture
    "policy_kwargs": {
        "net_arch": [256, 256],  # Two hidden layers with 256 units each
    },
    
    # Logging
    "verbose": 1,
    "tensorboard_log": str(LOGS_DIR),
}


# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================

CHECKPOINT_CONFIG: Dict[str, Any] = {
    # Save model at these timestep intervals
    "save_freq": 10_000,  # Save every 10k steps
    
    # Specific checkpoints for evolution video
    "untrained_name": "model_untrained",
    "midpoint_name": "model_midpoint",
    "final_name": "model_final",
    
    # Save paths
    "checkpoint_dir": MODELS_DIR,
}


# ============================================================================
# VIDEO RECORDING CONFIGURATION
# ============================================================================

VIDEO_CONFIG: Dict[str, Any] = {
    # Number of episodes to record per model
    "num_episodes": 5,
    
    # Video parameters
    "video_folder": str(VIDEOS_DIR),
    "name_prefix": "highway_evolution",
    "fps": 15,
    
    # Recording configuration
    "episode_trigger": lambda episode_id: True,  # Record all episodes
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVAL_CONFIG: Dict[str, Any] = {
    # Number of episodes for evaluation
    "n_eval_episodes": 10,
    
    # Deterministic policy for evaluation
    "deterministic": True,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path(name: str) -> Path:
    """
    Get the full path for a model checkpoint.
    
    Args:
        name: Name of the model (without extension)
        
    Returns:
        Full path to the model file
    """
    return MODELS_DIR / f"{name}.zip"


def print_config() -> None:
    """Print all configuration settings for verification."""
    print("=" * 80)
    print("HIGHWAY-ENV RL PROJECT CONFIGURATION")
    print("=" * 80)
    print(f"\nEnvironment: {ENV_NAME}")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Videos Directory: {VIDEOS_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"\nTotal Training Steps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Gamma (Discount): {TRAINING_CONFIG['gamma']}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
