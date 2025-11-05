for seed in 43 44; do
    job_name="nov5_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_01_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account stf --qos gpu-2080ti -sH 24 --mem 32 --cpus 5 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
