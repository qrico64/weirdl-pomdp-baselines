# python scripts/plot_csv_lib.py experiments/

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import csv
import sys
from pathlib import Path
import shutil
from typing import Optional

from scripts.delete_empties import env_steps_all_empty

def plot_envsteps_vs_eval_success(
    csv: str,
    out: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = (0.0, 1.05),
    marker: Optional[str] = "o",
    linewidth: float = 1.2,
    grid: bool = True,
    ax=None,
):
    """
    Plot z/env_steps vs metrics/success_rate_eval from a (possibly ragged) CSV.

    Parameters
    ----------
    csv : str
        Path to the CSV file.
    out : str | None, optional
        If provided, save the figure to this path (e.g., "plot.png"). If None,
        the plot is shown interactively (unless an Axes is supplied).
    title : str | None, optional
        Plot title. Defaults to "Eval Success Rate vs Environment Steps".
    xlim : (float, float) | None, optional
        Limits for the x-axis.
    ylim : (float, float) | None, optional
        Limits for the y-axis. Default is (0.0, 1.05).
    marker : str | None, optional
        Matplotlib marker for data points (e.g., "o", ".", None). Default "o".
    linewidth : float, optional
        Line width. Default 1.2.
    grid : bool, optional
        Whether to show a grid. Default True.
    ax : matplotlib.axes.Axes | None, optional
        If provided, draw on this Axes. Otherwise, create a new Figure/Axes.
    """

    if out is None:
        out = os.path.join(os.path.dirname(csv), "eval_success.png")

    ENV_STEPS_COL = "z/env_steps"
    SUCCESS_EVAL_COL = "metrics/success_rate_eval"

    # Read CSV with a tolerant parser (handles ragged rows like your sample)
    df = pd.read_csv(csv, engine="python")

    # Validate necessary columns
    missing = [c for c in (ENV_STEPS_COL, SUCCESS_EVAL_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only relevant columns, coerce to numeric, drop NaNs, and sort
    df = df[[ENV_STEPS_COL, SUCCESS_EVAL_COL]].copy()
    df[ENV_STEPS_COL] = pd.to_numeric(df[ENV_STEPS_COL], errors="coerce")
    df[SUCCESS_EVAL_COL] = pd.to_numeric(df[SUCCESS_EVAL_COL], errors="coerce")
    df = df.dropna(subset=[ENV_STEPS_COL, SUCCESS_EVAL_COL]).sort_values(ENV_STEPS_COL)

    if df.empty:
        raise ValueError("No valid rows after cleaning; nothing to plot.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True
    else:
        fig = ax.figure

    # Plot
    ax.plot(
        df[ENV_STEPS_COL].to_numpy(),
        df[SUCCESS_EVAL_COL].to_numpy(),
        marker=marker if marker else None,
        linewidth=linewidth,
    )

    # Labels & cosmetics
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Success Rate")
    ax.set_title(title or "Eval Success Rate vs Environment Steps")

    if grid:
        ax.grid(True, linestyle="--", alpha=0.4)

    if xlim is not None:
        ax.set_xlim(*xlim)
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

        if TARGET_FILE in filenames:
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

    print(f"[INFO] Found {len(candidates)} director{'y' if len(candidates)==1 else 'ies'} to remove:")
    for c in sorted(candidates):
        print(f" - {c}")

    for candidate in candidates:
        plot_envsteps_vs_eval_success(os.path.join(candidate, TARGET_FILE))



if __name__ == "__main__":
    main()

