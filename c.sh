python ./dtrain.py -j oct21_gpu_antdir_circle_down_up_16tasks_goal_uniform_02 --account cse --qos gpu-a100 -sH 24 --mem 64 --cpus 2 \
-- python policies/main.py \
    --cfg configs/meta/ant_dir/circle_down_up_16tasks_goal_uniform_02.yml \
