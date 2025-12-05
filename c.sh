for seed in 43 44; do
    job_name="dec5_antgoalposlinear_circle_3layers_combined_reward_x2_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account stf --qos gpu-2080ti -sH 24 --mem 45 --cpus 5 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
