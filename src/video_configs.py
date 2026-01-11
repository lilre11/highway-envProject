"""
Improved video configuration matching highway-env documentation style.
Creates videos with better visualization and camera settings.
"""

from pathlib import Path
from typing import Dict, Any

# Use this configuration for recording videos that match documentation style
IMPROVED_VIDEO_ENV_CONFIG: Dict[str, Any] = {
    # Observation (doesn't affect rendering, but keep for consistency)
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
    
    # Reward settings (for model evaluation)
    "collision_reward": -1.0,
    "reward_speed_range": [20, 30],
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "normalize_reward": True,
    
    # Simulation
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "vehicles_speed": 18,  # Slower traffic for overtaking
    
    # IMPROVED RENDERING SETTINGS
    "offscreen_rendering": True,  # Fixed: Prevent viewer conflicts
    "screen_width": 1200,  # Wider view (doubled from 600)
    "screen_height": 300,  # Taller view (doubled from 150)
    "centering_position": [0.25, 0.5],  # Show more ahead
    "scaling": 7.0,  # Adjust zoom level (higher = more zoomed in)
    "show_trajectories": False,  # Clean view without trajectory lines
    "render_agent": True,
}

# Alternative: Match documentation exactly
DOCUMENTATION_STYLE_CONFIG: Dict[str, Any] = {
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
    
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "initial_spacing": 2,
    "controlled_vehicles": 1,
    
    "collision_reward": -1.0,
    "reward_speed_range": [20, 30],
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "normalize_reward": True,
    
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "vehicles_speed": 18,
    
    # Documentation default rendering
    "offscreen_rendering": True,  # Fixed: Prevent viewer conflicts
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,  # Clean view
    "render_agent": True,
}

# Wide panoramic view
PANORAMIC_CONFIG: Dict[str, Any] = {
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
    
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "initial_spacing": 2,
    "controlled_vehicles": 1,
    
    "collision_reward": -1.0,
    "reward_speed_range": [20, 30],
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "normalize_reward": True,
    
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "vehicles_speed": 18,
    
    # Ultra-wide panoramic view
    "offscreen_rendering": True,  # Fixed: Prevent viewer conflicts
    "screen_width": 1600,  # Extra wide
    "screen_height": 200,
    "centering_position": [0.2, 0.5],  # Show even more ahead
    "scaling": 4.5,  # More zoomed out
    "show_trajectories": False,  # Clean view without trajectory lines
    "render_agent": True,
}
