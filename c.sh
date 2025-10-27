python ./dtrain.py -j oct27_nonparallel_antdir_circle_down_quarter_norm_5tasks_goal --account stf --qos gpu-l40 -sH 24 --mem 80 --cpus 20 \
-- python policies/main.py \
    --cfg flexible.yaml \
