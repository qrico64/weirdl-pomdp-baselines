# python ./dtrain.py -j oct16_antdir_circle_left_right_up_down_inftasks_numiter9k --account cse --qos gpu-2080ti -- python policies/main.py --cfg configs/meta/ant_dir/circle_left_right_up_down_inftasks_numiter9k.yml
import argparse
import os
from utils.helpers import today_str
import subprocess
import sys


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

    braket = "{PWD}"
    today = today_str()
    assert args.j.startswith(today), f"{today}"
    log_dir = f"experiments/{today}/{args.j}"
    res = f"""#!/bin/bash -l
#SBATCH --job-name={args.j}        # Job name
#SBATCH --output={log_dir}/%j_%x_out.txt        # Output file (%j = job ID)
#SBATCH --error={log_dir}/%j_%x_err.txt         # Error file
#SBATCH --time={str(args.sH).zfill(2)}:00:00            # Time limit (hh:mm:ss)
#SBATCH --partition=gpu            # Partition/queue name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task={args.cpus}          # CPUs per task
#SBATCH --gres=gpu:1               # GPUs per node (if needed)
#SBATCH --mem={args.mem}G                  # Memory per node
#SBATCH --partition={args.qos}        # Partition (queue) name
#SBATCH --account={args.account}         # Slurm account/project name

# Load environment
conda activate pomdp4
module load cuda/12.0
export PYTHONPATH=${braket}:$PYTHONPATH

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
    
    print(f"Submitting slurm file: {slurm_file}")

    sbatch_cmd = f"sbatch {slurm_file}"
    try:
        proc = subprocess.run(sbatch_cmd, shell=True, check=False, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    except FileNotFoundError:
        print("Error: 'sbatch' not found on PATH. Use --local to run without Slurm.", file=sys.stderr)
        return 127

if __name__ == '__main__':
    main()
