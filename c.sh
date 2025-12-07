for seed in 42 43; do
    job_name="dec6_nominalchanges2_circle_combined2_nominal_embed_16_nominal_trajectory__$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account stf --qos gpu-2080ti -sH 24 --mem 43 --cpus 5 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
