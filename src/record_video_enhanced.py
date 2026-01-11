"""
Enhanced video recording with multiple visualization styles.
Allows switching between different camera/rendering configurations.

Usage:
    python src/record_video_enhanced.py --style panoramic
    python src/record_video_enhanced.py --style documentation
    python src/record_video_enhanced.py --style improved
"""

import sys
from pathlib import Path
from typing import List, Tuple
import argparse

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
    CHECKPOINT_CONFIG,
    get_model_path,
    VIDEOS_DIR,
)
from src.video_configs import (
    IMPROVED_VIDEO_ENV_CONFIG,
    DOCUMENTATION_STYLE_CONFIG,
    PANORAMIC_CONFIG
)


def load_model(model_path: Path) -> DQN:
    """Load a trained DQN model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"[Load] Loading model: {model_path.name}")
    return DQN.load(model_path)


def record_episode(
    model: DQN,
    env: gym.Env,
    deterministic: bool = True
) -> Tuple[List[np.ndarray], float, int]:
    """Record a single episode."""
    frames: List[np.ndarray] = []
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    truncated = False

    while not (done or truncated):
        frame = env.render()
        frames.append(frame)
        
        action, _ = model.predict(obs, deterministic=deterministic)
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
    """Add text overlay to frames."""
    from PIL import Image, ImageDraw, ImageFont

    color_schemes = {
        "untrained": {"bg": (220, 53, 69), "text": (255, 255, 255)},
        "mid": {"bg": (255, 193, 7), "text": (33, 37, 41)},
        "final": {"bg": (40, 167, 69), "text": (255, 255, 255)},
    }
    
    colors = color_schemes.get(model_stage, {"bg": (33, 37, 41), "text": (255, 255, 255)})
    annotated_frames = []
    
    # Determine font sizes based on frame size
    frame_height = frames[0].shape[0]
    title_size = max(20, int(frame_height * 0.08))
    info_size = max(14, int(frame_height * 0.06))
    
    for frame_idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", title_size)
            info_font = ImageFont.truetype("arial.ttf", info_size)
        except (IOError, OSError):
            try:
                title_font = ImageFont.truetype("segoeui.ttf", title_size)
                info_font = ImageFont.truetype("segoeui.ttf", info_size)
            except (IOError, OSError):
                title_font = ImageFont.load_default()
                info_font = ImageFont.load_default()
        
        # Draw header
        header_height = int(frame_height * 0.25)
        draw.rectangle([(0, 0), (img.width, header_height)], fill=colors["bg"])
        
        # Title
        title_pos = (15, int(header_height * 0.1))
        draw.text(title_pos, text, fill=colors["text"], font=title_font)
        
        # Episode counter (top right)
        episode_text = f"Episode {episode_num}/{total_episodes}"
        episode_bbox = draw.textbbox((0, 0), episode_text, font=info_font)
        episode_width = episode_bbox[2] - episode_bbox[0]
        episode_pos = (img.width - episode_width - 15, int(header_height * 0.1))
        draw.text(episode_pos, episode_text, fill=colors["text"], font=info_font)
        
        # Reward info
        reward_text = f"Reward: {reward:.2f}"
        reward_pos = (15, int(header_height * 0.55))
        draw.text(reward_pos, reward_text, fill=colors["text"], font=info_font)
        
        # Frame counter (bottom right)
        frame_text = f"Frame {frame_idx + 1}/{len(frames)}"
        frame_bbox = draw.textbbox((0, 0), frame_text, font=info_font)
        frame_width = frame_bbox[2] - frame_bbox[0]
        frame_pos = (img.width - frame_width - 15, int(header_height * 0.55))
        draw.text(frame_pos, frame_text, fill=colors["text"], font=info_font)
        
        annotated_frames.append(np.array(img))
    
    return annotated_frames


def record_evolution_video(
    env_config: dict,
    num_episodes: int = 5,
    fps: int = 30,
    style_name: str = "default"
) -> None:
    """Record evolution video with specified visualization style."""
    
    print("\n" + "=" * 80)
    print(f"RECORDING MODEL EVOLUTION VIDEO - {style_name.upper()} STYLE")
    print("=" * 80)
    print(f"Resolution: {env_config['screen_width']}x{env_config['screen_height']}")
    print(f"Scaling: {env_config['scaling']}")
    print(f"Show Trajectories: {env_config['show_trajectories']}")
    print("-" * 80)

    checkpoints = [
        (CHECKPOINT_CONFIG["untrained_name"], "Untrained Agent"),
        (CHECKPOINT_CONFIG["midpoint_name"], "Mid-Training (50%)"),
        (CHECKPOINT_CONFIG["final_name"], "Fully Trained"),
    ]

    env = gym.make(ENV_NAME, config=env_config, render_mode="rgb_array")
    all_frames: List[np.ndarray] = []

    for checkpoint_name, display_text in checkpoints:
        model_path = get_model_path(checkpoint_name)
        
        try:
            model = load_model(model_path)
            print(f"\n[Recording] {display_text}")
            print("-" * 40)

            for episode in range(num_episodes):
                frames, reward, steps = record_episode(model, env, deterministic=True)
                
                if "Untrained" in display_text:
                    stage = "untrained"
                elif "Mid" in display_text:
                    stage = "mid"
                else:
                    stage = "final"
                
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

            # Removed black screen transitions for smoother video flow

        except FileNotFoundError as e:
            print(f"\n[Warning] {e}")
            print(f"[Warning] Skipping {display_text}")
            continue

    env.close()

    if not all_frames:
        print("\n[Error] No frames recorded!")
        return

    # Save video with style name
    output_path = VIDEOS_DIR / f"highway_evolution_{style_name}.mp4"
    print(f"\n[Saving] Generating video: {output_path}")
    print(f"[Saving] Total frames: {len(all_frames)} | FPS: {fps}")
    
    imageio.mimsave(output_path, all_frames, fps=fps)
    
    print(f"\n[Success] Video saved: {output_path}")
    print(f"[Success] Duration: {len(all_frames) / fps:.1f} seconds")
    print("=" * 80)


def main():
    """Main entry point with style selection."""
    parser = argparse.ArgumentParser(description='Record highway-env videos with different styles')
    parser.add_argument('--style', type=str, 
                       choices=['improved', 'documentation', 'panoramic'],
                       default='improved',
                       help='Visualization style (default: improved)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Episodes per model (default: 5)')
    parser.add_argument('--fps', type=int, default=15,
                       help='Video FPS (default: 15)')
    
    args = parser.parse_args()
    
    # Select configuration
    config_map = {
        'improved': IMPROVED_VIDEO_ENV_CONFIG,
        'documentation': DOCUMENTATION_STYLE_CONFIG,
        'panoramic': PANORAMIC_CONFIG
    }
    
    env_config = config_map[args.style]
    
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
        print("\n[Action Required] Run 'python src/train.py' first!")
        return

    record_evolution_video(
        env_config=env_config,
        num_episodes=args.episodes,
        fps=args.fps,
        style_name=args.style
    )


if __name__ == "__main__":
    main()
