# python ./dtrain.py -j oct16_antdir_circle_left_right_up_down_inftasks_numiter9k --account cse --qos gpu-2080ti -- python policies/main.py --cfg configs/meta/ant_dir/circle_left_right_up_down_inftasks_numiter9k.yml
import argparse
import os
from utils.helpers import today_str
import subprocess
import sys
from utils import infra
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--qos", required=True)
    ap.add_argument("-sH", default=24)
    ap.add_argument("--cpus", default=4)
    ap.add_argument("--mem", default=64)
    ap.add_argument("payload", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    git_root = infra.ensure_in_git_repo()
    base_branch_name = infra.ensure_not_detached(git_root)
    remote_name = infra.get_default_remote(git_root)
    today = today_str()
    assert args.j.startswith(today), f"{today}"
    log_dir = os.path.join("experiments", today, args.j)
    git_copy_dir = Path(os.path.join(log_dir, "weirdl-pomdp-baselines"))
    assert not os.path.exists(git_copy_dir), f"{git_copy_dir} already exists!"
    new_branch_name = args.j

    print("\n------ Copying Repository Into Run Dir ------")
    print(f"Source repo:   {git_root}")
    print(f"Base branch:   {base_branch_name}")
    print(f"Destination:   {git_copy_dir.absolute()}")
    print(f"Remote:        {remote_name}")
    # print(f"New branch:    {new_branch_name}")

    infra.copy_repo_withfilters(git_root, git_copy_dir, use_hardlinks=True, excludes=['experiments', 'slurm', 'logs'])
    new_dir_size = infra.format_size(infra.get_directory_size(git_copy_dir, False))
    print(f"Size:          {new_dir_size}.")
    print("------ Copied Repository Into Run Dir ------\n")

    # TODO: Not going to push the branch because I think that'll be too annoying.

    res = f"""#!/bin/bash -l
#SBATCH --job-name={args.j}        # Job name
#SBATCH --output={log_dir}/%j_%x_out.txt        # Output file (%j = job ID)
#SBATCH --error={log_dir}/%j_%x_err.txt         # Error file
#SBATCH --time={str(args.sH).zfill(2)}:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task={args.cpus}          # CPUs per task
#SBATCH --gres=gpu:1               # GPUs per node (if needed)
#SBATCH --mem={args.mem}G                  # Memory per node
#SBATCH --partition={args.qos}        # Partition (queue) name
#SBATCH --account={args.account}         # Slurm account/project name

# Load environment
cd {git_copy_dir.absolute()}
conda activate pomdp4
module load cuda/12.0
export PYTHONPATH=${'{PWD}'}:$PYTHONPATH

# Checks
echo
echo "Node: $(hostname)"
which python
python -V
python -c "import sys, pprint; pprint.pprint(sys.path[:5])"
echo
echo

# Run your program
{' '.join(args.payload[1:])} --j {args.j} --log_folder {log_dir}
"""
    os.makedirs(log_dir, exist_ok=False)
    slurm_file = os.path.join(log_dir, "c.slurm")
    with open(slurm_file, "w") as fi:
        fi.write(res)
    
    print(f"****** Submitting slurm file: {slurm_file} ******")

    sbatch_cmd = f"sbatch {slurm_file}"
    try:
        proc = subprocess.run(sbatch_cmd, shell=True, check=False, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        print(f"****** Submitted slurm file: {slurm_file} ******\n")
        return proc.returncode
    except FileNotFoundError:
        print("Error: 'sbatch' not found on PATH. Use --local to run without Slurm.", file=sys.stderr)
        return 127

if __name__ == '__main__':
    main()
