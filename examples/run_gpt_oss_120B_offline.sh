#!/bin/bash

export DATAPATH="/root/.cache/user_artifacts/gpt-oss-120b/dataset/perfect-blend-gptoss-20B-1M.jsonl"
export CACHE_DIR="/root/.cache"
export OUTPUT_DIR="/root/data/yilian/data/gpt_oss_120B_output"
export HIDDEN_STATES_DIR="/root/data/yilian/data/gpt_oss_120B"
export DRAFT_CONFIG="/sgl-workspace/SpecForge/configs/gpt-oss-120B-eagle3_4096.json"
export MODEL="openai/gpt-oss-120B"
export CHAT_TEMPLATE="gpt-oss"
export TP=4
export MAX_LENGTH=2048

# python scripts/view_data.py --data-path $HIDDEN_STATES_DIR/all_test/rows_0-5000/data_100.ckpt --tokenizer $MODEL_PATH
# python scripts/view_data.py --data-path $HIDDEN_STATES_DIR/all_train/rows_0-5000/data_100.ckpt --tokenizer $MODEL_PATH

export NUM_GPUS=4
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_offline.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $DRAFT_CONFIG \
    --train-data-path $DATAPATH \
    --train-hidden-states-path $HIDDEN_STATES_DIR \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --draft-global-batch-size 16 \
    --draft-micro-batch-size 1 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --dist-timeout=10 \
    --log-steps 1 \
    --report-to wandb \
    --wandb-project llama3-8b-eagle3 \
    --wandb-key 6d964382a153a908ea0c874f64309c6e1605412b \
    --wandb-name gpt-oss-4096
