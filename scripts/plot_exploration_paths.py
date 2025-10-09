# python scripts/plot_csv_lib.py experiments/

import os
import pandas as pd
import argparse
import csv
import sys
from pathlib import Path
import shutil
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from scripts.delete_empties import env_steps_all_empty
from envs.meta.make_env import make_env
from envs.meta.toy_navigation.point_robot import SparsePointEnv
from scripts.read_yaml import read_yaml_in_experiment
from scripts.render import *

def plot_envsteps_vs_eval_success(dir: str, env_type: str = 'circle_1_2'):
    log_file = os.path.join(dir, "experiment.log")
    config = read_yaml_in_experiment(dir)
    if isinstance(config['env']['goal_conditioning'], bool):
        config['env']['goal_conditioning'] = "yes" if config['env']['goal_conditioning'] else "no"
    with open(log_file, 'r') as file:
        lines = [line.rstrip('\n').strip() for line in file]
    
    env_step = -1
    movement = {}
    env: SparsePointEnv = make_env(config['env']['env_name'], config['env']['max_rollouts_per_task'], **config['env']).unwrapped
    for idx, line in enumerate(lines):
        if line.startswith("env steps "):
            assert line.replace('env steps ', '').isnumeric()
            env_step = int(line.replace('env steps ', ''))
        elif line.startswith("Task "):
            task_idx = int(line.split(' ')[1])
            task_pos = env.goals[task_idx]
            trajectories = eval(lines[idx + 1])
            if env_step not in movement:
                movement[env_step] = {}
            if task_idx not in movement:
                movement[env_step][task_idx] = {}
            movement[env_step][task_idx]['is_train'] = task_idx < config['env']['num_train_tasks']
            movement[env_step][task_idx]['task_pos'] = task_pos
            movement[env_step][task_idx]['trajectories'] = trajectories
    
    mins = env.goals.min(axis=0) - 0.5
    maxs = env.goals.max(axis=0) + 0.5
    H, W = 512, 512
    pixel_dims = np.array([W, H])

    last_ep = max(list(movement.keys()))

    eval_successes_by_position = {}
    for ep in movement.keys():
        for task_idx in movement[ep].keys():
            ret_num = 0
            ret_den = 0
            info = movement[ep][task_idx]
            for traj in info['trajectories']:
                ret_den += 1
                reward_probably = False
                for s in traj:
                    pos = np.array(s[:2])
                    reward_probably = np.linalg.norm(pos - info['task_pos']) <= env.goal_radius
                if reward_probably:
                    ret_num += 1
            if task_idx not in eval_successes_by_position:
                eval_successes_by_position[task_idx] = [0, 0]
            eval_successes_by_position[task_idx][0] += ret_num
            eval_successes_by_position[task_idx][1] += ret_den
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot
    xs = list(eval_successes_by_position.keys())
    ys = [eval_successes_by_position[i][0] for i in xs]
    ax.bar(xs, ys, color='steelblue')

    # Labels and title
    ax.set_xlabel("Task index")
    ax.set_ylabel(f"Number of Successes out of {eval_successes_by_position[0][1]}")
    ax.set_title("Eval Successes By Task")

    # Optional grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save to file
    fig.savefig(os.path.join(dir, "success_by_task.png"))
    print(os.path.join(dir, "success_by_task.png"))

    # frames = []
    # for task_idx in movement[last_ep]:
    #     info = movement[last_ep][task_idx]
    #     for traj in info['trajectories']:
    #         for s in traj:
    #             pos = np.array(s[:2])
    #             reward_probably = np.linalg.norm(pos - info['task_pos']) <= env.goal_radius
    #             pos_pixel = (pos - mins) / (maxs - mins) * pixel_dims
    #             goal_pixel = (info['task_pos'] - mins) / (maxs - mins) * pixel_dims
    #             goal_radius = env.goal_radius / (maxs - mins)[0] * pixel_dims[0]
    #             frame = render_circle_and_dot_rgb(pixel_dims, goal_pixel, goal_radius, 1, pos_pixel)
    #             frames.append(frame)
    #         frames += [frames[-1] for i in range(3)]
    #     frames += [frames[-1] for i in range(10)]
    # frames_stacked = stack_frames(frames)
    # video_path = write_video_mp4(frames_stacked, path=os.path.join(dir, "last_ep.mp4"))
    # print(video_path)






TARGET_FILE = "progress.csv"
TARGET_COL = "z/env_steps"

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
            plot_envsteps_vs_eval_success(candidate)
        except Exception:
            continue



if __name__ == "__main__":
    main()
    # plot_envsteps_vs_eval_success("experiments/oct7/30009919-rnn_8tasks_circle_1_2")

