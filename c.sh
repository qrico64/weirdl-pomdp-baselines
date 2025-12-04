for seed in 42; do
    job_name="dec4_antgoalposlinear_circle_3layers_combined_reward_x1_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account stf --qos gpu-2080ti -sH 24 --mem 45 --cpus 5 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
