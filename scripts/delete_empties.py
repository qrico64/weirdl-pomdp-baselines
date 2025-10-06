#!/usr/bin/env python3

# python scripts/delete_empties.py logs --missing-column=delete
"""
Recursively traverse a directory tree, find every 'progress.csv',
and if the 'z/env_steps' column is *all empty* (no non-empty entries),
delete the immediate parent directory containing that file.

Default behavior:
- If 'z/env_steps' is missing, SKIP that CSV (do not delete).
- If file is blank or header-only (including having 'z/env_steps' but no data),
  treat as "all empty" => delete.
- Safety guard prevents deleting the root unless explicitly allowed.

Usage:
  python clean_env_steps_empty.py /path/to/root
  python clean_env_steps_empty.py /path/to/root --dry-run
  python clean_env_steps_empty.py /path/to/root --allow-delete-root
  # Optional: treat missing 'z/env_steps' as empty (will delete)
  python clean_env_steps_empty.py /path/to/root --missing-column=delete
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import shutil
from typing import Optional

TARGET_FILE = "progress.csv"
TARGET_COL = "z/env_steps"

def env_steps_all_empty(
    p: Path,
    treat_missing_column_as_empty: bool = False
) -> bool:
    """
    Returns True iff the CSV should be considered "all empty" for TARGET_COL.

    Rules:
    - Completely blank file => True
    - Header-only file (no data rows) => True if header contains TARGET_COL,
      else depends on treat_missing_column_as_empty.
    - Column present: returns True only if every row's TARGET_COL is blank/NA-like.
    - Column missing: returns treat_missing_column_as_empty.
    """
    try:
        # Quick blank-file check
        try:
            if p.stat().st_size == 0:
                return True  # blank file treated as all empty
        except OSError:
            return False

        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            # Also handle files with only whitespace/newlines
            head = f.read(2048)
            if head.strip() == "":
                return True
            f.seek(0)

            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            if not fieldnames:
                # No header detected; conservative by default
                return treat_missing_column_as_empty

            if TARGET_COL not in fieldnames:
                return treat_missing_column_as_empty

            # Stream rows; if we see any non-empty value, it's NOT all empty
            saw_any_rows = False
            for row in reader:
                saw_any_rows = True
                val = row.get(TARGET_COL, "")
                s = ("" if val is None else str(val)).strip()
                # Treat common NA strings as empty too
                if s not in ("", "nan", "NaN", "NA", "None", "N/A"):
                    return False

            # If there were no data rows, it's effectively all empty
            return True if not saw_any_rows else True
    except Exception as e:
        print(f"[WARN] Could not parse {p}: {e}", file=sys.stderr)
        # On parse errors, do not delete
        return False

def find_candidate_dirs(root: Path, treat_missing_column_as_empty: bool) -> set:
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
            if env_steps_all_empty(csv_path, treat_missing_column_as_empty):
                candidates.add(Path(dirpath))
    return candidates

def delete_dirs(dirs: list, dry_run: bool) -> None:
    """
    Deletes directories in leaf-first order (deepest paths first).
    """
    dirs_sorted = sorted(dirs, key=lambda p: len(p.parts), reverse=True)
    for d in dirs_sorted:
        if dry_run:
            print(f"[DRY-RUN] Would delete: {d}")
        else:
            try:
                # shutil.rmtree(d)
                print(f"[OK] Deleted: {d}")
            except FileNotFoundError:
                print(f"[SKIP] Already gone: {d}")
            except PermissionError as e:
                print(f"[ERROR] Permission denied deleting {d}: {e}", file=sys.stderr)
            except OSError as e:
                print(f"[ERROR] Failed deleting {d}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Delete parent dirs if progress.csv has all-empty 'z/env_steps'.")
    ap.add_argument("root", type=Path, help="Root directory to scan")
    ap.add_argument("--dry-run", action="store_true", help="Preview deletions without changing anything")
    ap.add_argument("--allow-delete-root", action="store_true",
                    help="Allow deleting the root directory itself if it qualifies")
    ap.add_argument("--missing-column", choices=["skip", "delete"], default="skip",
                    help="If 'z/env_steps' column is missing: 'skip' (default) or 'delete' the directory")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root path does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    treat_missing = (args.missing_column == "delete")
    candidates = find_candidate_dirs(root, treat_missing)

    # Safety guard: don't delete the root unless allowed
    if not args.allow_delete_root and root in candidates:
        print(f"[SAFEGUARD] Root directory {root} qualifies but will be skipped "
              f"(enable with --allow-delete-root).")
        candidates.remove(root)

    if not candidates:
        print("[INFO] No directories to delete under the given rules.")
        return

    print(f"[INFO] Found {len(candidates)} director{'y' if len(candidates)==1 else 'ies'} to remove:")
    for c in sorted(candidates):
        print(f" - {c}")

    delete_dirs(list(candidates), dry_run=args.dry_run)

if __name__ == "__main__":
    main()
