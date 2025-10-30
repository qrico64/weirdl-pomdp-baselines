# python scripts/plot_csv_lib.py experiments/oct14/ --column all -f 1

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from utils import helpers

from scripts.delete_empties import env_steps_all_empty

FILE_NAMES = {
    'metrics/success_rate_train': ['train_success.png', 'Train Success Rate', (0, 1.05)],
    'metrics/success_rate_eval': ['eval_success.png', 'Eval Success Rate', (0, 1.05)],
    'metrics/return_eval_total': ['eval_return.png', 'Eval Return', None],
    'metrics/return_train_total': ['train_return.png', 'Train Return', None],
}
ENV_STEPS_COL = "z/env_steps"

def compile_data(csv: str, column):
    # Read CSV with a tolerant parser (handles ragged rows like your sample)
    df = pd.read_csv(csv, engine="python")

    # Validate necessary columns
    missing = [c for c in (ENV_STEPS_COL, column) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only relevant columns, coerce to numeric, drop NaNs, and sort
    df = df[[ENV_STEPS_COL, column]].copy()
    df[ENV_STEPS_COL] = pd.to_numeric(df[ENV_STEPS_COL], errors="coerce")
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[ENV_STEPS_COL, column]).sort_values(ENV_STEPS_COL)

    if df.empty:
        raise ValueError("No valid rows after cleaning; nothing to plot.")

    if df[ENV_STEPS_COL].to_numpy().shape[0] > 40:
        indices = np.linspace(0, df[ENV_STEPS_COL].to_numpy().shape[0], 40, endpoint=False).round().astype(np.int32)
        return df[ENV_STEPS_COL].to_numpy()[indices], df[column].to_numpy()[indices]
    else:
        return df[ENV_STEPS_COL].to_numpy(), df[column].to_numpy()



def plot_data(csv: str, xs, ys, column='metrics/success_rate_eval'):
    out = os.path.join(os.path.dirname(csv), FILE_NAMES[column][0])
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot
    ax.plot(xs, ys, marker="o", linewidth=1.2)

    # Labels & cosmetics
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(FILE_NAMES[column][1])
    ax.set_title(f"Environment Steps vs. {FILE_NAMES[column][1]}")
    ax.grid(True, linestyle="--", alpha=0.4)

    ylim = FILE_NAMES[column][2]
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Nicely format large x-axis ticks (e.g., 1,200,000)
    try:
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
    except Exception:
        pass

    fig.tight_layout()

    fig.savefig(out, dpi=200)
    print(out)


def plot_comparison(out: str, parents: list, column, labels: list = None, title: str = None):
    assert out.startswith(f"viz/{helpers.today_str()}/")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    assert os.path.basename(out).endswith(".png")

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, parent in enumerate(parents):
        # Check if parent is a list of paths (for aggregation) or a single path
        if isinstance(parent, list):
            # Aggregate multiple runs: compute mean and std
            all_xs = []
            all_ys = []

            for p in parent:
                csv = os.path.join(p, "progress.csv")
                assert os.path.exists(csv), f"{csv}"
                xs, ys = compile_data(csv, column)
                all_xs.append(xs)
                all_ys.append(ys)

            # Find common x-axis (use the shortest one or interpolate)
            # For simplicity, we'll use interpolation to align all runs to a common x-axis
            min_x = max([xs[0] for xs in all_xs])
            max_x = min([xs[-1] for xs in all_xs])
            common_xs = np.linspace(min_x, max_x, 100)

            # Interpolate all runs to common x-axis
            interpolated_ys = []
            for xs, ys in zip(all_xs, all_ys):
                interp_ys = np.interp(common_xs, xs, ys)
                interpolated_ys.append(interp_ys)

            # Compute mean and std
            interpolated_ys = np.array(interpolated_ys)
            mean_ys = np.mean(interpolated_ys, axis=0)
            std_ys = np.std(interpolated_ys, axis=0)

            label = labels[idx] if labels is not None else f"Aggregated {idx}"
            line = ax.plot(common_xs, mean_ys, linewidth=1.5, label=label)
            ax.fill_between(common_xs, mean_ys - std_ys, mean_ys + std_ys, alpha=0.3, color=line[0].get_color())
        else:
            # Single run: plot as before
            csv = os.path.join(parent, "progress.csv")
            assert os.path.exists(csv), f"{csv}"
            xs, ys = compile_data(csv, column)
            ax.plot(xs, ys, marker="o", linewidth=1.2, label=labels[idx] if labels is not None else os.path.basename(parent))

    # Labels & cosmetics
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(FILE_NAMES[column][1])
    ax.set_title(title or f"Environment Steps vs. {FILE_NAMES[column][1]}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ylim = FILE_NAMES[column][2]
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Nicely format large x-axis ticks (e.g., 1,200,000)
    try:
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
    except Exception:
        pass

    fig.tight_layout()

    fig.savefig(out, dpi=200)
    print(out)

    with open(os.path.join(os.path.dirname(out), os.path.basename(out)[:-4] + ".log"), "w") as fi:
        fi.write(f"{str(parents)}\n{str(labels)}")
    print()






TARGET_FILE = "progress.csv"
TARGET_COL = "z/env_steps"

def find_candidate_dirs(root: Path, column: str, force: bool = False) -> set:
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

        if not force and FILE_NAMES[column][0] in filenames:
            continue
        if TARGET_FILE in filenames:
            csv_path = Path(dirpath) / TARGET_FILE
            if not env_steps_all_empty(csv_path, True):
                candidates.add(Path(dirpath))
    return candidates

def main():
    ap = argparse.ArgumentParser(description="Delete parent dirs if progress.csv has all-empty 'z/env_steps'.")
    ap.add_argument("root", type=Path, help="Root directory to scan")
    ap.add_argument("--column", choices=list(FILE_NAMES.keys()) + ['all'], default='all', help='Which side are you on?')
    ap.add_argument("-f", choices=['0', '1'])
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root path does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    
    if args.column == 'all':
        if args.f is None:
            args.f = '1'
        assert args.f == '1', f"Must regenerate all plots."

    candidates = find_candidate_dirs(root, column=args.column, force=args.f == '1')

    if not candidates:
        print("[INFO] No directories to delete under the given rules.")
        return

    print(f"[INFO] Found {len(candidates)} director{'y' if len(candidates)==1 else 'ies'} to plot:")
    for c in sorted(candidates):
        print(f" - {c}")

    for candidate in candidates:
        if args.column == "all":
            columns = list(FILE_NAMES.keys())
        else:
            columns = [args.column]
        csv = os.path.join(candidate, TARGET_FILE)
        for column in columns:
            try:
                xs, ys = compile_data(csv, column)
                plot_data(csv, xs, ys, column)
            except:
                pass



if __name__ == "__main__":
    main()
    # plot_comparison("viz/oct27/debug_variance_train.png", [
    #     "experiments/oct27/oct27_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal",
    #     "experiments/oct27/oct27_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal_debug_1",
    #     "experiments/oct27/oct27_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal_debug_2",
    #     "experiments/oct27/oct27_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal_debug_3_cpu2",
    #     "experiments/oct27/oct27_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal_debug_4_cpu2_numrollouts16",
    #     "experiments/oct24/oct24_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal",
    #     # "experiments/oct27/oct27_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal_normal_01_cpu11",
    #     # "experiments/oct27/oct27_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal_normal_04_cpu11",
    # ], column="metrics/return_train_total", labels=[
    #     "goal",
    #     "goal_debug_1",
    #     "goal_debug_2",
    #     "goal_debug_3_cpu2",
    #     "goal_debug_4_cpu2_numrollouts16",
    #     "oct24",
    #     # "goal_normal_01_cpu11",
    #     # "goal_normal_04_cpu11",
    # ], title="Debugging Variance (Train)")

