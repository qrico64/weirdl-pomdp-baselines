# python scripts/plot_csv_lib.py experiments/

import os
import argparse
import sys
from pathlib import Path
import numpy as np
import tqdm

from scripts.delete_empties import env_steps_all_empty
from envs.meta.make_env import make_env
from scripts.read_yaml import read_yaml_in_experiment
from scripts.render import stack_frames, write_video_mp4, render_circle, render_dot, render_line

def plot_trajectories_pointenv(dir: str):
    log_file = os.path.join(dir, "experiment.log")
    config = read_yaml_in_experiment(dir)
    if 'goal_conditioning' not in config['env']:
        config['env']['goal_conditioning'] = 'no'
    if isinstance(config['env']['goal_conditioning'], bool):
        config['env']['goal_conditioning'] = "yes" if config['env']['goal_conditioning'] else "no"
    with open(log_file, 'r') as file:
        lines = [line.rstrip('\n').strip() for line in file]
    
    render_log_file = open(os.path.join(dir, "last_ep.log"), "w")
    
    env_step = -1
    movement = {}
    env = make_env(config['env']['env_name'], config['env']['max_rollouts_per_task'], **config['env']).unwrapped
    for idx, line in enumerate(lines):
        if line.startswith("env steps "):
            assert line.replace('env steps ', '').isnumeric()
            env_step = int(line.replace('env steps ', ''))
        elif line.startswith("Task "):
            task_idx = int(line.split(' ')[1])
            task_pos = env.goals[task_idx]
            if config['env']['env_name'] == "AntDir-v0":
                assert isinstance(task_pos, float)
                assert task_pos == eval(line.split(' ')[-1][:-1]), f"{task_pos} != {line.split(' ')[-1][:-1]}"
                # task_pos = np.array([np.cos(task_pos), np.sin(task_pos)])
            trajectories = eval(lines[idx + 1])
            trajectories = [np.array(trajectory) for trajectory in trajectories]
            assert all(trajectory.ndim == 2 and trajectory.shape[1] == 2 for trajectory in trajectories), f"{[trajectory.shape for trajectory in trajectories]}"
            if env_step not in movement:
                movement[env_step] = {}
            if task_idx not in movement:
                movement[env_step][task_idx] = {}
            movement[env_step][task_idx]['is_train'] = task_idx < config['env']['num_train_tasks']
            movement[env_step][task_idx]['task_pos'] = task_pos
            movement[env_step][task_idx]['trajectories'] = trajectories

            if config['env']['env_name'] == "AntDir-v0":
                mins = np.array([trajectory.min(axis=0) for trajectory in trajectories]).min(axis=0)
                maxs = np.array([trajectory.max(axis=0) for trajectory in trajectories]).max(axis=0)
            else:
                mins = env.goals.min(axis=0) - 0.5
                maxs = env.goals.max(axis=0) + 0.5
            mids = (mins + maxs) / 2
            lens = maxs - mins
            mins, maxs = mids - lens.max(), mids + lens.max()
            movement[env_step][task_idx]['mins'] = mins
            movement[env_step][task_idx]['maxs'] = maxs

    last_ep = max(list(movement.keys()))

    H, W = 512, 512
    pixel_dims = np.array([W, H])

    frames = []
    for task_idx in tqdm.tqdm(list(movement[last_ep].keys())):
        info = movement[last_ep][task_idx]
        for traj in info['trajectories']:
            past_frame = np.zeros((W, H, 3))
            render_log_file.write(f"{info['task_pos']}\n")
            for s in traj:
                mins = info['mins']
                maxs = info['maxs']

                frame = past_frame.copy()
                origin_pixel = ((0 - mins) / (maxs - mins) * pixel_dims).astype(dtype=np.int32)
                origin_pixel = [W - origin_pixel[1] - 1, origin_pixel[0]]
                frame[origin_pixel[0]-1 : origin_pixel[0]+2, :] = (255, 255, 255)
                frame[:, origin_pixel[1]-1 : origin_pixel[1]+2] = (255, 255, 255)

                pos = np.array(s[:2])
                pos_pixel = (pos - mins) / (maxs - mins) * pixel_dims
                pos_pixel = [W - pos_pixel[1] - 1, pos_pixel[0]]

                if config['env']['env_name'] == "AntDir-v0":
                    frame = render_line(frame, pos_pixel, info['task_pos'], 1, (255, 255, 255))
                else:
                    goal_pixel = (info['task_pos'] - mins) / (maxs - mins) * pixel_dims
                    goal_radius = env.goal_radius / (maxs - mins)[0] * pixel_dims[0]
                    goal_pixel = [W - goal_pixel[1] - 1, goal_pixel[0]]
                    frame = render_circle(frame, goal_pixel, goal_radius, 2, (255, 255, 255))

                frame = render_dot(frame, pos_pixel, 5)
                past_frame = render_dot(past_frame, pos_pixel, 2, color=(0, 255, 0))
                
                frames.append(frame.astype(np.uint8))
            frames += [frames[-1] for i in range(5)]
        frames += [frames[-1] for i in range(10)]
    print("Compressing frames...")
    frames_stacked = stack_frames(frames)
    print("Writing video...")
    video_path = write_video_mp4(frames_stacked, path=os.path.join(dir, "last_ep.mp4"))
    print(video_path)






TARGET_FILE = "progress.csv"

def find_candidate_dirs(root: Path) -> set:
    """
    Walks the tree and collects parent directories of progress.csv files where
    TARGET_COL is all empty under the configured rules.
    """
    candidates: set = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip symlinked dirs to avoid surprises
        try:
            if Path(dirpath).is_symlink():
                continue
        except OSError:
            continue

        if TARGET_FILE in filenames and "last_ep.mp4" not in filenames:
            csv_path = Path(dirpath) / TARGET_FILE
            if not env_steps_all_empty(csv_path, True):
                candidates.add(Path(dirpath))
    return candidates

def main():
    ap = argparse.ArgumentParser(description="Delete parent dirs if progress.csv has all-empty 'z/env_steps'.")
    ap.add_argument("root", type=Path, help="Root directory to scan")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root path does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    candidates = find_candidate_dirs(root)

    if not candidates:
        print("[INFO] No directories to delete under the given rules.")
        return

    print(f"[INFO] Found {len(candidates)} director{'y' if len(candidates)==1 else 'ies'} to plot:")
    for c in sorted(candidates):
        print(f" - {c}")

    for candidate in candidates:
        print(f"Encoding {candidate}")
        try:
            plot_trajectories_pointenv(candidate)
        except Exception:
            continue



if __name__ == "__main__":
    # main()
    plot_trajectories_pointenv("experiments/oct15/30107935-oct15_antdir_circle_16tasks_down_up_goal")

