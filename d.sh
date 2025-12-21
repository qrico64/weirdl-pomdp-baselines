CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python policies/main.py \
    --cfg configs/antgoal.yaml \
    -j debug
