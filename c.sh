python ./dtrain.py -j oct20_antdir_circle_inftasks_numiter9k_goal_random_02 --account weirdlab --qos gpu-l40 -sH 24 --mem 16 \
-- python policies/main.py \
    --cfg configs/meta/ant_dir/circle_inftasks_numiter9k_goal_random_02.yml \
