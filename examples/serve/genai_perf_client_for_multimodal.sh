#! /usr/bin/env bash

export ISL=(64)
export OSL=(64)

export CONCURRENCY=(1 50 300)

# Main loop
for isl in "${ISL[@]}"; do
    for osl in "${OSL[@]}"; do
        for concurrency in "${CONCURRENCY[@]}"; do
            echo "Running with ISL=$isl, OSL=$osl, CONCURRENCY=$concurrency"

            genai-perf profile \
                -m Qwen2-VL-7B-Instruct \
                --tokenizer /code/llm-models/Qwen2-VL-7B-Instruct \
                --endpoint-type multimodal \
                --random-seed 123 \
                --image-width-mean 512 \
                --image-height-mean 512 \
                --image-format png \
                --synthetic-input-tokens-mean "$isl" \
                --synthetic-input-tokens-stddev 0 \
                --output-tokens-mean "$osl" \
                --output-tokens-stddev 0 \
                --warmup-request-count 2 \
                --profile-export-file "ISL_${isl}_OSL_${osl}_CONCURRENCY_${concurrency}.json" \
                --url localhost:8000 \
                --num-prompts "$concurrency" \
                --concurrency "$concurrency" \
                --request-count $((concurrency * 5)) \
                --extra-inputs "max_tokens:$osl" \
                --extra-inputs "min_tokens:$osl" \
                --extra-inputs "ignore_eos:true" \
                --streaming \
                -- \
                -v \
                --max-threads 1
        done
    done
done
