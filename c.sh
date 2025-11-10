for seed in 40; do
    job_name="nov9_transformer_4layers_48emb_antdir_circle_norm_inftasks_goal_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account cse --qos gpu-l40s -sH 24 --mem 200 --cpus 30 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
