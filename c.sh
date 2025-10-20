python ./dtrain.py -j oct20_gpu_antdir_circle_16tasks_down_up_goal_random_0 --account stf --qos gpu-2080ti -sH 24 --mem 16 \
-- python policies/main.py \
    --cfg configs/meta/ant_dir/circle_16tasks_down_up_goal_random_0.yml \
