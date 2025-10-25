python ./dtrain.py -j oct24_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal_relative --account stf --qos gpu-2080ti -sH 24 --mem 20 --cpus 1 \
-- python policies/main.py \
    --cfg flexible.yaml \
