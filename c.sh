python ./dtrain.py -j oct19_antdir_circle_left_right_up_down_inftasks_numiter9k_2 --account stf --qos gpu-2080ti -sH 24 \
-- python policies/main.py \
    --cfg configs/meta/ant_dir/circle_left_right_up_down_inftasks_numiter9k.yml \
