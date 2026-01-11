"""
Video recording script for Highway-Env trained models.
Generates evolution video showing Untrained → Mid-Training → Fully-Trained performance.

Usage:
    python src/record_video.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import highway_env
import imageio
import numpy as np
from stable_baselines3 import DQN

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ENV_NAME,
    ENV_CONFIG,
    VIDEO_CONFIG,
    CHECKPOINT_CONFIG,
    get_model_path,
    VIDEOS_DIR,
)


def load_model(model_path: Path) -> DQN:
    """
    Load a trained DQN model from disk.

    Args:
        model_path: Path to the model checkpoint

    Returns:
        Loaded DQN model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please run 'python src/train.py' first to generate model checkpoints."
        )
    
    print(f"[Load] Loading model: {model_path.name}")
    model = DQN.load(model_path)
    return model


def record_episode(
    model: DQN,
    env: gym.Env,
    deterministic: bool = True
) -> Tuple[List[np.ndarray], float, int]:
    """
    Record a single episode with the given model.

    Args:
        model: DQN model to evaluate
        env: Gymnasium environment
        deterministic: Whether to use deterministic policy

    Returns:
        Tuple of (frames, total_reward, total_steps)
    """
    frames: List[np.ndarray] = []
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    truncated = False

    while not (done or truncated):
        # Render and capture frame
        frame = env.render()
        frames.append(frame)

        # Get action from model
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    return frames, total_reward, steps


def add_text_to_frames(
    frames: List[np.ndarray],
    text: str,
    reward: float,
    model_stage: str,
    episode_num: int,
    total_episodes: int
) -> List[np.ndarray]:
    """
    Add enhanced text overlay to frames using PIL with colored styling.

    Args:
        frames: List of image frames
        text: Main text to overlay
        reward: Episode reward
        model_stage: Stage of training (untrained, mid, final)
        episode_num: Current episode number
        total_episodes: Total episodes for this model

    Returns:
        List of frames with enhanced text overlay
    """
    from PIL import Image, ImageDraw, ImageFont

    # Color scheme based on model stage
    color_schemes = {
        "untrained": {"bg": (220, 53, 69), "text": (255, 255, 255)},      # Red
        "mid": {"bg": (255, 193, 7), "text": (33, 37, 41)},               # Yellow/Orange
        "final": {"bg": (40, 167, 69), "text": (255, 255, 255)},          # Green
    }
    
    colors = color_schemes.get(model_stage, {"bg": (33, 37, 41), "text": (255, 255, 255)})

    annotated_frames = []
    
    for frame_idx, frame in enumerate(frames):
        # Convert numpy array to PIL Image
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Try to use a better font
        try:
            title_font = ImageFont.truetype("arial.ttf", 28)
            info_font = ImageFont.truetype("arial.ttf", 20)
        except (IOError, OSError):
            try:
                title_font = ImageFont.truetype("segoeui.ttf", 28)
                info_font = ImageFont.truetype("segoeui.ttf", 20)
            except (IOError, OSError):
                title_font = ImageFont.load_default()
                info_font = ImageFont.load_default()
        
        # Draw header bar with colored background
        header_height = 80
        draw.rectangle([(0, 0), (img.width, header_height)], fill=colors["bg"])
        
        # Draw title text
        title_text = text
        title_pos = (15, 10)
        draw.text(title_pos, title_text, fill=colors["text"], font=title_font)
        
        # Draw episode counter
        episode_text = f"Episode {episode_num}/{total_episodes}"
        episode_bbox = draw.textbbox((0, 0), episode_text, font=info_font)
        episode_width = episode_bbox[2] - episode_bbox[0]
        episode_pos = (img.width - episode_width - 15, 10)
        draw.text(episode_pos, episode_text, fill=colors["text"], font=info_font)
        
        # Draw reward info
        reward_text = f"Reward: {reward:.2f}"
        reward_pos = (15, 45)
        draw.text(reward_pos, reward_text, fill=colors["text"], font=info_font)
        
        # Draw frame counter
        frame_text = f"Frame {frame_idx + 1}/{len(frames)}"
        frame_bbox = draw.textbbox((0, 0), frame_text, font=info_font)
        frame_width = frame_bbox[2] - frame_bbox[0]
        frame_pos = (img.width - frame_width - 15, 45)
        draw.text(frame_pos, frame_text, fill=colors["text"], font=info_font)
        
        # Convert back to numpy array
        annotated_frames.append(np.array(img))
    
    return annotated_frames


def record_model_evolution(
    num_episodes: int = 3,
    fps: int = 30
) -> None:
    """
    Record evolution video showing progression from untrained to fully trained.

    Args:
        num_episodes: Number of episodes to record per model state
        fps: Frames per second for output video
    """
    print("\n" + "=" * 80)
    print("RECORDING MODEL EVOLUTION VIDEO")
    print("=" * 80)

    # Model checkpoints to record
    checkpoints = [
        (CHECKPOINT_CONFIG["untrained_name"], "Untrained Agent"),
        (CHECKPOINT_CONFIG["midpoint_name"], "Mid-Training (50%)"),
        (CHECKPOINT_CONFIG["final_name"], "Fully Trained"),
    ]

    # Create environment with rendering enabled
    env_config = ENV_CONFIG.copy()
    env_config["offscreen_rendering"] = False
    env_config["render_agent"] = True  # Ensure agent vehicle is visible
    env_config["render_agent"] = True  # Ensure agent vehicle is visible
    env = gym.make(ENV_NAME, config=env_config, render_mode="rgb_array")

    all_frames: List[np.ndarray] = []

    # Record episodes for each checkpoint
    for checkpoint_name, display_text in checkpoints:
        model_path = get_model_path(checkpoint_name)
        
        try:
            model = load_model(model_path)
            print(f"\n[Recording] {display_text}")
            print("-" * 40)

            for episode in range(num_episodes):
                frames, reward, steps = record_episode(model, env, deterministic=True)
                
                # Determine model stage for color coding
                if "Untrained" in display_text:
                    stage = "untrained"
                elif "Mid" in display_text:
                    stage = "mid"
                else:
                    stage = "final"
                
                # Add enhanced text overlay to frames
                annotated_frames = add_text_to_frames(
                    frames=frames,
                    text=display_text,
                    reward=reward,
                    model_stage=stage,
                    episode_num=episode + 1,
                    total_episodes=num_episodes
                )
                
                all_frames.extend(annotated_frames)
                
                print(f"  Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {reward:.2f} | Steps: {steps} | "
                      f"Frames: {len(frames)}")

            # Add separator frames with transition effect
            if checkpoint_name != CHECKPOINT_CONFIG["final_name"]:
                # Create a smooth fade-to-black transition (0.5 seconds)
                fade_frames = fps // 2
                for i in range(fade_frames):
                    alpha = i / fade_frames
                    fade_frame = (all_frames[-1] * (1 - alpha)).astype(np.uint8)
                    all_frames.append(fade_frame)
                
                # Pure black separator (1 second)
                separator = np.zeros_like(all_frames[-1])
                all_frames.extend([separator] * fps)
                
                # Fade from black (0.5 seconds)
                for i in range(fade_frames):
                    alpha = i / fade_frames
                    fade_frame = (separator * (1 - alpha)).astype(np.uint8)
                    all_frames.append(fade_frame)

        except FileNotFoundError as e:
            print(f"\n[Warning] {e}")
            print(f"[Warning] Skipping {display_text}")
            continue

    env.close()

    if not all_frames:
        print("\n[Error] No frames recorded. Ensure models exist and run training first.")
        return

    # Save video
    output_path = VIDEOS_DIR / f"{VIDEO_CONFIG['name_prefix']}_evolution.mp4"
    print(f"\n[Saving] Generating video: {output_path}")
    print(f"[Saving] Total frames: {len(all_frames)} | FPS: {fps}")
    
    imageio.mimsave(output_path, all_frames, fps=fps)
    
    print(f"\n[Success] Video saved: {output_path}")
    print(f"[Success] Duration: {len(all_frames) / fps:.1f} seconds")
    print("=" * 80)


def record_single_model(
    model_name: str,
    num_episodes: int = 5,
    fps: int = 30
) -> None:
    """
    Record video for a single model checkpoint.

    Args:
        model_name: Name of the model checkpoint (without extension)
        num_episodes: Number of episodes to record
        fps: Frames per second for output video
    """
    print(f"\n[Recording] Single model: {model_name}")
    
    # Load model
    model_path = get_model_path(model_name)
    model = load_model(model_path)

    # Create environment
    env_config = ENV_CONFIG.copy()
    env_config["offscreen_rendering"] = False
    env = gym.make(ENV_NAME, config=env_config, render_mode="rgb_array")

    all_frames: List[np.ndarray] = []

    for episode in range(num_episodes):
        frames, reward, steps = record_episode(model, env, deterministic=True)
        
        # Add enhanced text overlay
        annotated_frames = add_text_to_frames(
            frames=frames,
            text=f"Model: {model_name}",
            reward=reward,
            model_stage="final",
            episode_num=episode + 1,
            total_episodes=num_episodes
        )
        all_frames.extend(annotated_frames)
        
        print(f"  Episode {episode + 1}/{num_episodes} | "
              f"Reward: {reward:.2f} | Steps: {steps}")

    env.close()

    # Save video
    output_path = VIDEOS_DIR / f"{model_name}_recording.mp4"
    print(f"\n[Saving] Video: {output_path}")
    imageio.mimsave(output_path, all_frames, fps=fps)
    print(f"[Success] Video saved successfully")


def main() -> None:
    """Main video recording pipeline."""
    print("\n" + "=" * 80)
    print("HIGHWAY-ENV VIDEO RECORDING")
    print("=" * 80)

    # Check if models exist
    required_models = [
        CHECKPOINT_CONFIG["untrained_name"],
        CHECKPOINT_CONFIG["midpoint_name"],
        CHECKPOINT_CONFIG["final_name"],
    ]

    missing_models = []
    for model_name in required_models:
        if not get_model_path(model_name).exists():
            missing_models.append(model_name)

    if missing_models:
        print("\n[Warning] Missing model checkpoints:")
        for model in missing_models:
            print(f"  - {model}")
        print("\n[Action Required] Run 'python src/train.py' first to generate models.")
        return

    # Record evolution video
    record_model_evolution(
        num_episodes=VIDEO_CONFIG["num_episodes"],
        fps=VIDEO_CONFIG["fps"]
    )

    print("\n[Complete] Video recording finished successfully!")
    print(f"[Output] Check the '{VIDEOS_DIR}' folder for videos.")


if __name__ == "__main__":
    main()
