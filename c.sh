for seed in 41; do
    job_name="nov9_fixedtype_attempt6_$((seed - 39))"
    python ./dtrain.py -j "$job_name" --account weirdlab --qos gpu-a40 -sH 24 --mem 160 --cpus 11 \
    -- python policies/main.py \
        --cfg flexible.yaml \
        --seed "$seed"
done
