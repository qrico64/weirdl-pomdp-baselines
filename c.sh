python ./dtrain.py -j oct27_nonparallel_antdir_circle_down_quarter_norm_16tasks_goal_relative_normal_02 --account stf --qos ckpt -sH 24 --mem 128 --cpus 12 \
-- python policies/main.py \
    --cfg flexible.yaml \
