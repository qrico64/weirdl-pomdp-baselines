import os
from pathlib import Path

def delete_buffers(root_dir: str, dry_run: bool = False):
    root_path = Path(root_dir).resolve()
    total_size = 0
    deleted_files = []

    for path in root_path.rglob("buffer_*.pt"):
        if path.is_file() and path.parent.name == "save":
            try:
                size = path.stat().st_size
                total_size += size
                deleted_files.append((path, size))
                if not dry_run:
                    path.unlink()
                    print(f"Deleted: {path} ({size / 1e6:.2f} MB)")
                else:
                    print(f"Found: {path} ({size / 1e6:.2f} MB)")
            except Exception as e:
                print(f"Failed to process {path}: {e}")

    if not deleted_files:
        print(f"\nNo matching files found under {root_path}")
        return

    print("\n" + ("✅ DRY RUN COMPLETE:" if dry_run else "✅ DELETION COMPLETE:"))
    print(f"  Total files {'to be deleted' if dry_run else 'deleted'}: {len(deleted_files)}")
    print(f"  Total storage {'to be freed' if dry_run else 'freed'}: "
          f"{total_size / 1e6:.2f} MB ({total_size / 1e9:.2f} GB)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively delete files named buffer_*.pt whose parent folder is 'save'."
    )
    parser.add_argument("root_dir", type=str, help="Root directory to start searching from")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List matching files and total size without deleting them"
    )

    args = parser.parse_args()
    delete_buffers(args.root_dir, dry_run=args.dry_run)
