for seed in 44; do
    job_name="dec2_py310_peg_fixed_combined1_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account weirdlab --qos gpu-l40 -sH 24 --mem 180 --cpus 16 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
