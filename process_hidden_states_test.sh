#!/bin/bash

# Script to process first 10 items from perfect-blend dataset for hidden states generation
# Based on settings from run_gpt_oss_120b_eagle3_online.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR

# Configuration from GPT-OSS example
MODEL_PATH="openai/gpt-oss-120b"
DATA_PATH="/mnt/raid0/home/yilian/data/perfect-blend-425k.jsonl"
CACHE_DIR="$ROOT_DIR/cache"
OUTPUT_DIR="$ROOT_DIR/cache/hidden_states/gpt-oss-120b-test"
MAX_LENGTH=2048
CHAT_TEMPLATE="gpt-oss"
NUM_SAMPLES=10
TP_SIZE=1  # Adjust based on your GPU setup

# Create necessary directories
mkdir -p $OUTPUT_DIR
mkdir -p $CACHE_DIR

echo "Processing first $NUM_SAMPLES samples from $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Cache directory: $CACHE_DIR"

# Run hidden states generation
if [ $TP_SIZE -gt 1 ]; then
    echo "Running with tensor parallelism (TP=$TP_SIZE)"
    torchrun \
        --standalone \
        --nproc_per_node $TP_SIZE \
        $ROOT_DIR/scripts/prepare_hidden_states.py \
        --generator-backend huggingface \
        --model-path $MODEL_PATH \
        --data-path $DATA_PATH \
        --output-path $OUTPUT_DIR \
        --cache-dir $CACHE_DIR \
        --chat-template $CHAT_TEMPLATE \
        --max-length $MAX_LENGTH \
        --num-samples $NUM_SAMPLES \
        --tp-size $TP_SIZE \
        --batch-size 1 \
        --enable-aux-hidden-states \
        --trust-remote-code \
        --seed 42
else
    echo "Running on single GPU"
    python $ROOT_DIR/scripts/prepare_hidden_states.py \
        --generator-backend huggingface \
        --model-path $MODEL_PATH \
        --data-path $DATA_PATH \
        --output-path $OUTPUT_DIR \
        --cache-dir $CACHE_DIR \
        --chat-template $CHAT_TEMPLATE \
        --max-length $MAX_LENGTH \
        --num-samples $NUM_SAMPLES \
        --tp-size 1 \
        --batch-size 1 \
        --enable-aux-hidden-states \
        --trust-remote-code \
        --seed 42
fi

echo "Hidden states generation completed!"
echo "Output saved to: $OUTPUT_DIR"