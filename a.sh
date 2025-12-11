conda activate pomdp4
module load cuda/12.0
export PYTHONPATH=${PWD}:$PYTHONPATH
eval "$(ssh-agent -s)"
ssh-add
