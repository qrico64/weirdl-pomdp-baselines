#!/usr/bin/env python3
"""
snapshot_push.py

Create a copy of the current Git repository (including .git so local/uncommitted changes
come along), then—in the COPY ONLY—create a new branch forked from the current branch,
commit all local changes, and push that new branch to the remote.

This script does NOT modify your original working directory.

Usage:
    python snapshot_push.py --dest /path/to/new/location
    [--remote origin] [--branch-name <name>] [--hardlink]
    [--commit-message "Snapshot commit"]

Python >= 3.8
"""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# -------------------------------
# Utilities
# -------------------------------

class CmdError(RuntimeError):
    pass


def run_cmd(args, cwd: Optional[Path] = None, capture: bool = True) -> str:
    """Run a command, return stdout (str) or raise CmdError."""
    try:
        res = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
        return (res.stdout or "").strip()
    except subprocess.CalledProcessError as e:
        msg = f"Command failed: {' '.join(args)}\n"
        if e.stdout:
            msg += f"STDOUT:\n{e.stdout}\n"
        if e.stderr:
            msg += f"STDERR:\n{e.stderr}\n"
        raise CmdError(msg) from e


def ensure_in_git_repo() -> Path:
    """Return repo root if inside a Git repo; else raise."""
    try:
        root = run_cmd(["git", "rev-parse", "--show-toplevel"])
    except CmdError:
        raise RuntimeError("Not inside a Git repository (no .git found upward).")
    return Path(root)


def ensure_not_detached(repo_dir: Path) -> str:
    """Ensure HEAD is on a branch; return branch name."""
    branch = run_cmd(["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repo_dir)
    if not branch:
        raise RuntimeError("HEAD appears detached. Please check out a branch first.")
    return branch


def get_default_remote(repo_dir: Path) -> str:
    """Return 'origin' if present, else the first configured remote, else raise."""
    remotes = run_cmd(["git", "remote"], cwd=repo_dir).splitlines()
    remotes = [r.strip() for r in remotes if r.strip()]
    if not remotes:
        raise RuntimeError("No Git remotes configured in this repository.")
    return "origin" if "origin" in remotes else remotes[0]


def suggest_branch_name(base_branch: str) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"snapshot/{base_branch}/{ts}"


def get_directory_size(directory: Path, follow_symlinks: bool = False) -> int:
    """
    Calculate the total disk usage of a directory in bytes.

    Args:
        directory: Path to the directory to measure
        follow_symlinks: If True, follow symbolic links and count their targets.
                        If False (default), count only the symlink itself.

    Returns:
        Total size in bytes

    Raises:
        OSError: If directory doesn't exist or is not accessible
    """
    total_size = 0

    if not directory.exists():
        raise OSError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise OSError(f"Path is not a directory: {directory}")

    for entry in directory.rglob('*'):
        try:
            if entry.is_symlink() and not follow_symlinks:
                # Count the symlink itself, not its target
                total_size += entry.lstat().st_size
            elif entry.is_file():
                # Use stat() which follows symlinks by default, or lstat() which doesn't
                stat_func = entry.stat if follow_symlinks else entry.lstat
                total_size += stat_func().st_size
        except (OSError, PermissionError):
            # Skip files we can't access
            continue

    return total_size


def format_size(size_bytes: int) -> str:
    """
    Format a size in bytes to a human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable string (e.g., "1.5 GB", "256.0 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def copy_repo_withfilters(src_root: Path, dst_root: Path, use_hardlinks: bool, excludes: list = []) -> None:
    """
    Copy entire repo directory tree including .git so working copy state (unstaged changes,
    untracked files, index, etc.) is preserved. This is the simplest and most accurate
    way to mirror local changes without touching the original repo.
    """
    if dst_root.exists() and any(dst_root.iterdir()):
        raise RuntimeError(f"Destination '{dst_root}' exists and is not empty.")
    
    def _ignore_function(current: str, entries: list):
        if Path(current).resolve() == Path(src_root).resolve():
            # ignored = [name for name in entries if name in excludes]
            return set(excludes)
        return set()

    # Prefer hardlinks for speed/space if requested and same filesystem.
    # Fallback to regular copies if hardlinking fails for specific files.
    def _copy_function(src, dst, *, follow_symlinks=True):
        if use_hardlinks:
            try:
                return os.link(src, dst)
            except OSError:
                # Fallback to copy2 if hardlink not possible (e.g., cross-filesystem)
                return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
        else:
            return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)

    shutil.copytree(
        src_root,
        dst_root,
        copy_function=_copy_function,
        symlinks=True,            # preserve symlinks as symlinks
        dirs_exist_ok=False,       # require destination to not exist or be empty
        ignore=_ignore_function,
    )


# -------------------------------
# Core workflow (performed only in the COPY)
# -------------------------------

def create_branch_commit_and_push(
    copy_dir: Path,
    remote: str,
    new_branch: str,
    commit_message: str,
) -> None:
    """In the copied repo, create new branch, commit all changes, and push."""
    # Ensure we're on the original base branch in the copy
    base_branch = ensure_not_detached(copy_dir)

    # Create the new branch from current branch tip (fork point)
    run_cmd(["git", "switch", "-c", new_branch], cwd=copy_dir)

    # Stage everything (tracked changes + untracked files)
    run_cmd(["git", "add", "-A"], cwd=copy_dir)

    # If there is nothing to commit, we still push the new branch to share the base state
    status = run_cmd(["git", "status", "--porcelain"], cwd=copy_dir)
    if status.strip():
        # There are changes to commit
        run_cmd(["git", "commit", "-m", commit_message], cwd=copy_dir)
    else:
        # No changes—still ensure branch exists on remote
        pass

    # Push the new branch upstream
    run_cmd(["git", "push", "-u", remote, new_branch], cwd=copy_dir)

    # Print a helpful summary
    remote_url = run_cmd(["git", "remote", "get-url", remote], cwd=copy_dir)
    print(f"\n✓ Snapshot branch pushed.")
    print(f"  Base branch:   {base_branch}")
    print(f"  New branch:    {new_branch}")
    print(f"  Remote:        {remote} ({remote_url})")
    print(f"  Repo copy at:  {copy_dir}")


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy current Git repo (including .git), then in the copy create a new branch, commit local changes, and push to remote."
    )
    p.add_argument(
        "--dest",
        required=True,
        help="Destination directory for the repo copy (will be created; must be empty or non-existent).",
    )
    p.add_argument(
        "--remote",
        default=None,
        help="Remote name to push to (default: 'origin' if present, else first configured remote).",
    )
    p.add_argument(
        "--branch-name",
        default=None,
        help="Name for the new branch (default: snapshot/<current-branch>/<timestamp>).",
    )
    p.add_argument(
        "--commit-message",
        default=None,
        help='Commit message for the snapshot (default: "Snapshot from <base> at <timestamp>").',
    )
    p.add_argument(
        "--hardlink",
        action="store_true",
        help="Attempt to hardlink files when copying (fast & space-efficient; only works on same filesystem).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Validate environment and inputs
    src_root = ensure_in_git_repo()
    base_branch = ensure_not_detached(src_root)

    dst_root = Path(args.dest).resolve()
    if dst_root.exists() and any(dst_root.iterdir()):
        print(f"ERROR: Destination '{dst_root}' exists and is not empty.", file=sys.stderr)
        sys.exit(2)

    remote = args.remote or get_default_remote(src_root)
    new_branch = args.branch_name or suggest_branch_name(base_branch)

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = args.commit_message or f"Snapshot from {base_branch} at {ts}"

    print(f"Source repo:   {src_root}")
    print(f"Base branch:   {base_branch}")
    print(f"Destination:   {dst_root}")
    print(f"Remote:        {remote}")
    print(f"New branch:    {new_branch}")
    print(f"Hardlink:      {args.hardlink}")
    print("Copying repository (including .git)...")

    # Copy the repo including .git to preserve local changes
    copy_repo_withfilters(src_root, dst_root, use_hardlinks=args.hardlink)

    # In the COPY: create branch, commit, and push
    try:
        create_branch_commit_and_push(
            copy_dir=dst_root,
            remote=remote,
            new_branch=new_branch,
            commit_message=commit_message,
        )
    except CmdError as e:
        # Clean error with context; keep the copy around for inspection
        print("\nERROR while creating/pushing snapshot branch:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
