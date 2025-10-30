python ./dtrain.py -j oct30_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_04_1 --account cse --qos gpu-l40s -sH 24 --mem 200 --cpus 2 \
-- python policies/main.py \
    --cfg flexible.yaml \
