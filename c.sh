for seed in 44; do
    job_name="dec21_hypothesis1_markov_circle_1_2_goal_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account weirdlab --qos gpu-l40s -sH 24 --mem 187 --cpus 16 \
    -- python policies/main.py \
        --cfg configs/antgoal.yaml \
        --seed "$seed"
done
