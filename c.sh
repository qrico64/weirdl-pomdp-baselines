for seed in 41; do
    job_name="nov26_logging_circle_3layers_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account cse --qos gpu-l40s -sH 24 --mem 128 --cpus 40 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
