python ./dtrain.py -j oct29_nominalinevals_antdir_circle_down_quarter_norm_16tasks_goal_nominal --account weirdlab --qos gpu-l40s -sH 24 --mem 256 --cpus 19 \
-- python policies/main.py \
    --cfg flexible.yaml \
