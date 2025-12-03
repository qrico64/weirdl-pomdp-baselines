#!/usr/bin/env python3
"""
Test script for MetaWorld peg-insertion environment.
Runs random actions and renders a video to video.mp4.

Usage:
    source a.sh  # Activate conda environment first
    python envs/meta/mujoco/peg_insertion_test.py
"""


# ============================================================
# PEG AND TARGET POSITION BOUNDS
# ============================================================

# Peg Position Bounds:
#   Low:  [0.   0.5  0.02]
#   High: [0.2  0.7  0.02]

# Default Peg Initial Position: [0.18491799 0.66787545 0.02      ]
# Current Peg Position: [0.18491799 0.66787545 0.02      ]

# Target Position Bounds:
#   Low:  [-0.32   0.4    0.129]
#   High: [-0.22   0.7    0.131]

# Current Target Position: [-0.28711311  0.4484228   0.12945647]
# ============================================================

import os

# Configure rendering for headless server (before importing any rendering libs)
os.environ['MUJOCO_GL'] = 'egl'  # Use OSMesa for software rendering

import metaworld
import gymnasium as gym
import numpy as np
import imageio


def get_position_bounds(env):
    """
    Get the bounds for peg and target positions from the MetaWorld environment.

    Args:
        env: The gymnasium environment (must be MetaWorld peg-insertion)

    Returns:
        dict: Dictionary containing bounds information
    """
    mw_env = env.unwrapped

    bounds = {}

    # Get object (peg) position bounds
    if hasattr(mw_env, 'obj_init_pos'):
        bounds['peg_init_pos'] = mw_env.obj_init_pos

    if hasattr(mw_env, '_random_reset_space'):
        obj_space = mw_env._random_reset_space
        bounds['peg_bounds'] = {
            'low': obj_space.low[:3],  # First 3 dimensions are usually xyz
            'high': obj_space.high[:3]
        }

    # Get goal/target position bounds
    if hasattr(mw_env, '_target_pos'):
        bounds['current_target_pos'] = mw_env._target_pos

    if hasattr(mw_env, 'goal_space'):
        goal_space = mw_env.goal_space
        bounds['target_bounds'] = {
            'low': goal_space.low,
            'high': goal_space.high
        }

    # Get current object position from MuJoCo
    if hasattr(mw_env, 'get_body_com'):
        try:
            current_peg_pos = mw_env.get_body_com('peg')
            bounds['current_peg_pos'] = current_peg_pos
        except:
            pass

    return bounds


def print_position_bounds(env):
    """
    Print the position bounds in a readable format.

    Args:
        env: The gymnasium environment (must be MetaWorld peg-insertion)
    """
    print("\n" + "="*60)
    print("PEG AND TARGET POSITION BOUNDS")
    print("="*60)

    bounds = get_position_bounds(env)

    if 'peg_bounds' in bounds:
        print("\nPeg Position Bounds:")
        print(f"  Low:  {bounds['peg_bounds']['low']}")
        print(f"  High: {bounds['peg_bounds']['high']}")

    if 'peg_init_pos' in bounds:
        print(f"\nDefault Peg Initial Position: {bounds['peg_init_pos']}")

    if 'current_peg_pos' in bounds:
        print(f"Current Peg Position: {bounds['current_peg_pos']}")

    if 'target_bounds' in bounds:
        print("\nTarget Position Bounds:")
        print(f"  Low:  {bounds['target_bounds']['low']}")
        print(f"  High: {bounds['target_bounds']['high']}")

    if 'current_target_pos' in bounds:
        print(f"\nCurrent Target Position: {bounds['current_target_pos']}")

    print("="*60 + "\n")

    return bounds


def set_peg_and_target_positions(env, peg_pos=None, target_pos=None):
    """
    Set the peg and target positions in the MetaWorld peg-insertion environment.

    Args:
        env: The gymnasium environment (must be MetaWorld peg-insertion)
        peg_pos: 3D position [x, y, z] for the peg (default: None, keeps random)
        target_pos: 3D position [x, y, z] for the target hole (default: None, keeps random)
    """
    # Access the underlying MetaWorld environment
    mw_env = env.unwrapped

    # Set peg position (this is the object the robot needs to grasp)
    if peg_pos is not None:
        peg_pos = np.array(peg_pos, dtype=np.float32)
        # Set the peg's initial position
        mw_env._set_obj_xyz(peg_pos)
        print(f"  Set peg position to: {peg_pos}")

    # Set target position (the hole where the peg should be inserted)
    if target_pos is not None:
        target_pos = np.array(target_pos, dtype=np.float32)
        # Set the goal position
        mw_env._target_pos = target_pos
        mw_env.goal = target_pos
        print(f"  Set target position to: {target_pos}")


def main():
    # Create the peg-insertion environment from MetaWorld
    # Using 'peg-insert-side-v3' task
    print("Creating MetaWorld peg-insertion environment...")
    env = gym.make('Meta-World/MT1', env_name='peg-insert-side-v3', render_mode='rgb_array')

    # Parameters
    num_episodes = 1
    max_steps_per_episode = 200
    fps = 30

    # ============================================================
    # CONFIGURE PEG AND TARGET POSITIONS HERE (in code)
    # ============================================================
    # Set to None to use random positions, or provide [x, y, z] coordinates
    # Example positions (adjust based on your workspace bounds):
    # PEG_POSITION = [0.0, 0.6, 0.02]      # Peg starting position
    # TARGET_POSITION = [0.1, 0.8, 0.15]   # Target hole position

    PEG_POSITION = None      # Set to [x, y, z] or None for random
    TARGET_POSITION = None   # Set to [x, y, z] or None for random
    # ============================================================

    # Storage for video frames
    frames = []

    print(f"Running {num_episodes} episode(s) with up to {max_steps_per_episode} steps each...")

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Reset the environment
        observation, info = env.reset()

        # Print position bounds (useful for determining valid positions)
        if episode == 0:
            print_position_bounds(env)

        # Set custom peg and target positions (if specified)
        set_peg_and_target_positions(env, peg_pos=PEG_POSITION, target_pos=TARGET_POSITION)

        print(f"  Initial observation shape: {observation.shape}")
        print(f"  Action space: {env.action_space}")

        episode_reward = 0

        for step in range(max_steps_per_episode):
            # Render and capture frame
            frame = env.render()
            frames.append(frame)

            # Sample a random action
            action = env.action_space.sample()

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if step % 50 == 0:
                print(f"  Step {step}/{max_steps_per_episode}, Reward: {reward:.4f}")

            # Check if episode is done
            done = terminated or truncated
            if done:
                print(f"  Episode finished at step {step + 1}")
                print(f"  Terminated: {terminated}, Truncated: {truncated}")
                break

        print(f"  Total episode reward: {episode_reward:.4f}")

    # Close the environment
    env.close()

    # Save video
    output_path = 'video.mp4'
    print(f"\nSaving video with {len(frames)} frames to {output_path}...")

    try:
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved successfully to {output_path}")
        print(f"Video specs: {len(frames)} frames, {fps} FPS, duration ~{len(frames)/fps:.2f}s")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Attempting alternative method with ffmpeg writer...")
        try:
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"Video saved successfully to {output_path}")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("You may need to install ffmpeg: conda install -c conda-forge ffmpeg")

    print("\nDone!")


if __name__ == "__main__":
    main()
