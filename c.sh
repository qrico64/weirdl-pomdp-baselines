for seed in 44; do
    job_name="nov29_logging_circle_3layers_separate_src_key_padding_mask_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account cse --qos gpu-l40s -sH 24 --mem 160 --cpus 12 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
