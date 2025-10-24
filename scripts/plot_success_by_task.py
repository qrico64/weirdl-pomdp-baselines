# python scripts/plot_csv_lib.py experiments/

import os
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.delete_empties import env_steps_all_empty
from envs.meta.make_env import make_env
from envs.meta.toy_navigation.point_robot import SparsePointEnv
from scripts.read_yaml import read_yaml_in_experiment

def plot_success_rate_by_task(dir: str, column="return", after=0):
    log_file = os.path.join(dir, "experiment.log")
    config = read_yaml_in_experiment(dir)
    if isinstance(config['env']['goal_conditioning'], bool):
        config['env']['goal_conditioning'] = "yes" if config['env']['goal_conditioning'] else "no"
    with open(log_file, 'r') as file:
        lines = [line.rstrip('\n').strip() for line in file]
    
    env_step = -1
    env: SparsePointEnv = make_env(config['env']['env_name'], config['env']['max_rollouts_per_task'], **config['env']).unwrapped
    eval_successes_by_position = {}
    for idx, line in enumerate(lines):
        if line.startswith("env steps "):
            assert line.replace('env steps ', '').isnumeric()
            env_step = int(line.replace('env steps ', ''))
        elif line.startswith("Task "):
            if env_step < after:
                continue
            task_idx = int(line.split(' ')[1])
            task_pos = env.goals[task_idx]
            trajectories = eval(lines[idx + 1])

            if column == "return":
                assert lines[idx + 3].startswith('[') and lines[idx + 3].strip().endswith(']')
                returns = eval(lines[idx + 3].strip())
                if task_idx not in eval_successes_by_position:
                    eval_successes_by_position[task_idx] = [0, 0]
                eval_successes_by_position[task_idx][0] += returns[0]
                eval_successes_by_position[task_idx][1] += 1
            else:
                ret_num = 0
                ret_den = 0
                for traj in trajectories:
                    ret_den += 1
                    reward_probably = False
                    for s in traj:
                        pos = np.array(s[:2])
                        reward_probably = np.linalg.norm(pos - task_pos) <= env.goal_radius
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
    if column == "return":
        ys = [eval_successes_by_position[i][0] / eval_successes_by_position[i][1] for i in xs]
    else:
        ys = [eval_successes_by_position[i][0] for i in xs]
    ax.bar(xs, ys, color='steelblue')

    # Labels and title
    jobname = Path(dir).name
    ax.set_xlabel("Task index")
    if column == "return":
        ax.set_ylabel(f"Returns")
        ax.set_title(f"Returns By Task ({jobname})")
    else:
        ax.set_ylabel(f"Number of Successes out of {eval_successes_by_position[0][1]}")
        ax.set_title(f"Eval Successes By Task ({jobname})")

    # Optional grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save to file
    if column == "return":
        outputdir = os.path.join(dir, "return_by_task.png")
    else:
        outputdir = os.path.join(dir, "success_by_task.png")
    fig.savefig(outputdir)
    print(outputdir)



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
    ap.add_argument("--after", default=0, type=float)
    ap.add_argument("--column", choices=["return", "success"], type=str)
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
            plot_success_rate_by_task(candidate, column=args.column, after=args.after)
        except Exception as e:
            print(e)
            continue



if __name__ == "__main__":
    main()
    # plot_success_rate_by_task("experiments/oct7/30009919-rnn_8tasks_circle_1_2")
