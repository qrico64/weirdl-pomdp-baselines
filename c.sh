for seed in 41; do
    job_name="nov6_transformers_antdir_circle_norm_16tasks_goal_transformers_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account weirdlab --qos gpu-l40 -sH 24 --mem 200 --cpus 20 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
