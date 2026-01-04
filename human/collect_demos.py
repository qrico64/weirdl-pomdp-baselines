import sys
import numpy as np
import torch
from ruamel.yaml import YAML
from absl import flags
import os
import time
import matplotlib
matplotlib.use("MacOSX")  # or "QtAgg"
import matplotlib.pyplot as plt
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer
from PIL import Image
import inspect

from utils import system
import torchkit.pytorch_utils as ptu
from envs.meta.make_env import make_env
from envs.meta.wrappers import VariBadWrapper

FLAGS = flags.FLAGS
flags.DEFINE_string("cfg", None, "path to configuration file")
flags.DEFINE_string("j", None, "Job name.")
flags.mark_flag_as_required("cfg")
flags.mark_flag_as_required("j")


camera_ids = ["behindGripper", "gripperPOV", "topview"]


def data_collection_cycle(
        env: VariBadWrapper,
        num_rollouts: int,
        save_file_header: str,
    ):
    """
    Collect human demonstrations using keyboard input.

    Args:
        env: The wrapped environment
        num_rollouts: Number of trajectories to collect
        save_file_header: Prefix for saved files
    """

    # Initialize replay buffer
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_trajectory_len = env.env._max_episode_steps

    replay_buffer = SeqReplayBuffer(
        max_replay_buffer_size=num_rollouts * max_trajectory_len,
        observation_dim=obs_dim,
        action_dim=act_dim,
        sampled_seq_len=max_trajectory_len,
        sample_weight_baseline=0.0,
        observation_type=env.observation_space.dtype,
    )

    # Keyboard state
    key_state = {
        'w': False, 'a': False, 's': False, 'd': False,
        'up': False, 'down': False, ' ': False,
        'b': False,
    }

    def on_key_press(event):
        key = event.key.lower() if event.key != ' ' else ' '
        if key in key_state:
            key_state[key] = True

    def on_key_release(event):
        key = event.key.lower() if event.key != ' ' else ' '
        if key in key_state:
            key_state[key] = False

    def get_action_from_keyboard():
        """Convert keyboard state to 4D action.
        Controls rotated 90° counterclockwise to match wrist camera view:
        - W: +X (right in wrist view)
        - S: -X (left in wrist view)
        - A: +Y (forward in wrist view)
        - D: -Y (backward in wrist view)
        """
        action = np.zeros(4)

        # X-axis control (WASD rotated: W=+1, S=-1)
        if key_state['w']:
            action[0] += 1.0
        if key_state['s']:
            action[0] -= 1.0

        # Y-axis control (WASD rotated: A=+1, D=-1)
        if key_state['a']:
            action[1] += 1.0
        if key_state['d']:
            action[1] -= 1.0

        # Z-axis control (Up/Down arrows)
        if key_state['up']:
            action[2] += 1.0
        if key_state['down']:
            action[2] -= 1.0

        # 4th dimension: -1 if space pressed, 1 otherwise
        action[3] = -1.0 if key_state[' '] else 1.0

        return action

    # Setup figure with subplots for each camera
    num_cameras = len(camera_ids)
    fig, axes = plt.subplots(1, num_cameras, figsize=(8 * num_cameras, 8))

    # Ensure axes is always a list even with single camera
    if num_cameras == 1:
        axes = [axes]

    # Initialize image plots (will be updated in place for better performance)
    img_plots = [None] * num_cameras

    # Disable default matplotlib keyboard shortcuts to prevent conflicts with WASD controls
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    plt.ion()
    plt.show(block=False)

    print("Starting data collection...")
    print("Controls (rotated to match wrist camera view):")
    print("  W: +X (right in wrist view)")
    print("  S: -X (left in wrist view)")
    print("  A: +Y (forward in wrist view)")
    print("  D: -Y (backward in wrist view)")
    print("  Up/Down arrows: Z movement")
    print("  Space: Toggle 4th dimension (-1 when pressed, +1 when released)")
    print("  Close window or press Ctrl+C to stop")

    for rollout_idx in range(num_rollouts):
        print(f"\n=== Starting Rollout {rollout_idx + 1}/{num_rollouts} ===")

        # Reset environment
        obs, info = env.reset()
        done = False

        # Storage for current trajectory
        observations = []
        actions = []
        rewards = []
        terminals = []
        next_observations = []

        step_count = 0
        total_reward = 0.0
        success_reached = False

        # 5Hz operation: 0.2 seconds per step
        target_dt = 0.2

        stats = [[] for i in range(4)]
        while not done:
            # Get action from keyboard first
            action = get_action_from_keyboard()

            t0 = time.perf_counter_ns()
            # Step environment
            next_obs, reward, done, info = env.step(action)
            stats[0].append(time.perf_counter_ns() - t0)
            t0 = time.perf_counter_ns()

            # Check for success (following policies/learner.py:892-893)
            if "is_goal_state" in info and info["is_goal_state"]:
                success_reached = True

            # Store transition
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)
            next_observations.append(next_obs)

            # Update state
            obs = next_obs
            total_reward += reward
            step_count += 1

            # Stop trajectory if success is reached
            if success_reached:
                print(f"Success reached at step {step_count}!")
                done = True

            stats[1].append(time.perf_counter_ns() - t0)
            t0 = time.perf_counter_ns()
            # Render and display (optimized)
            try:
                # Render each camera
                camera_images = []
                for idx, camera_name in enumerate(camera_ids):
                    img = env.unwrapped.render(camera_name)
                    camera_images.append(img)
                stats[2].append(time.perf_counter_ns() - t0)
                t0 = time.perf_counter_ns()

                # Update images efficiently (no clear())
                if img_plots[0] is None:
                    # First frame - create the image plots
                    for idx, (ax, img, camera_name) in enumerate(zip(axes, camera_images, camera_ids)):
                        img_plots[idx] = ax.imshow(img)
                        ax.set_title(camera_name)
                        ax.axis('off')
                else:
                    # Update existing image data (much faster)
                    for idx, img in enumerate(camera_images):
                        img_plots[idx].set_data(img)

                # Update title
                fig.suptitle(f"Rollout {rollout_idx + 1}/{num_rollouts}, Step {step_count}, Reward: {reward:.2f}, Return: {total_reward:.2f}")

                # Efficient canvas update
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)  # Minimal pause just to process events

            except Exception as e:
                print(f"Warning: Could not render: {e}")
            stats[3].append(time.perf_counter_ns() - t0)
            t0 = time.perf_counter_ns()

            # Check if figure is closed
            if not plt.fignum_exists(fig.number):
                print("Window closed. Stopping data collection.")
                done = True
                break

            if len(stats[0]) % 20 == 1:
                print(np.array(stats).mean(axis=1) / 1000)

        # Convert lists to numpy arrays
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        terminals = np.array(terminals).reshape(-1, 1)
        next_observations = np.array(next_observations)

        # Add episode to replay buffer
        episode_info = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_observations=next_observations,
            nominals=np.ones_like(rewards, dtype=np.int64),  # All steps are nominal
        )
        replay_buffer.add_episode(**episode_info)

        print(f"Rollout {rollout_idx + 1} complete: {step_count} steps, total reward: {total_reward:.2f}")

        # Check if we should continue
        if not plt.fignum_exists(fig.number):
            print("Window closed. Stopping data collection.")
            break

    # Save replay buffer
    save_path = f"{save_file_header}_demos.pt"
    torch.save(replay_buffer, save_path)
    # print(f"\nSaved {replay_buffer.num_complete_episodes()} trajectories to {save_path}")
    print(f"Total transitions: {replay_buffer.size()}")

    plt.close('all')


def main():
    FLAGS(sys.argv)

    # Load configuration
    yaml = YAML()
    v = yaml.load(open(FLAGS.cfg))

    # System setup: seed
    seed = v["seed"]
    system.reproduce(seed)

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    ptu.set_gpu_mode(torch.cuda.is_available() and v["cuda"] >= 0, v["cuda"])

    # Extract environment configuration
    env_args = v["env"]
    env = make_env(
        env_id=env_args["env_name"],
        episodes_per_task=1,
        seed=seed,
        num_train_tasks=env_args.get("num_train_tasks", None),
        num_eval_tasks=env_args.get("num_eval_tasks", None),
    )

    # Default to 10 rollouts if not specified in config
    num_rollouts = v.get("num_demo_rollouts", 10)

    # Create save file header from job name
    save_file_header = f"demos/{FLAGS.j}"
    os.makedirs("demos", exist_ok=True)

    data_collection_cycle(env, num_rollouts, save_file_header)


if __name__ == "__main__":
    main()
