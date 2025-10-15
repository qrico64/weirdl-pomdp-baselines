#!/bin/bash -l
#SBATCH --job-name=oct14_circle_windfixed_4tasks_dense_goal        # Job name
#SBATCH --output=slurm/oct15/%j_%x_out.txt        # Output file (%j = job ID)
#SBATCH --error=slurm/oct15/%j_%x_err.txt         # Error file
#SBATCH --time=5:00:00            # Time limit (hh:mm:ss)
#SBATCH --partition=gpu            # Partition/queue name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=4          # CPUs per task
#SBATCH --gres=gpu:1               # GPUs per node (if needed)
#SBATCH --mem=16G                  # Memory per node
#SBATCH --partition=gpu-2080ti        # Partition (queue) name
#SBATCH --account=stf         # Slurm account/project name

# Load environment
conda activate pomdp4
module load cuda/12.0
export PYTHONPATH=${PWD}:$PYTHONPATH

# Checks
echo
echo "Node: $(hostname)"
which python
python -V
python -c "import sys, pprint; pprint.pprint(sys.path[:5])"
echo
echo

# Run your program
python policies/main.py --cfg configs/meta/circlev2/circle_windfixed_4tasks_dense_goal.yml --j oct14_circle_windfixed_4tasks_dense_goal
