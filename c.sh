for seed in 40 41; do
    job_name="nov6_transformers_antdir_circle_down_up_norm_4tasks_nogoal_transformers_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account stf --qos gpu-2080ti -sH 24 --mem 12 --cpus 2 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
