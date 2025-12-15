for seed in 44 45; do
    job_name="dec3_py310_peg_random_combined1_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account weirdlab --qos gpu-l40s -sH 24 --mem 128 --cpus 5 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
