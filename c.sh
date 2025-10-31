python ./dtrain.py -j oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_3 --account cse --qos gpu-l40s -sH 24 --mem 256 --cpus 32 \
-- python policies/main.py \
    --cfg flexible.yaml \
