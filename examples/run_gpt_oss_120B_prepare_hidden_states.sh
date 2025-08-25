export DATAPATH="/root/.cache/user_artifacts/gpt-oss-20b-1M/dataset/perfect-blend-gptoss-20B-1M.jsonl"
export CACHE_DIR="/root/.cache"
export OUTPUT_PATH="/root/data/yilian/data/gpt_oss_120B"
export MODEL="openai/gpt-oss-120B"
export CHAT_TEMPLATE="gpt-oss"
export TP=4

uv run -- torchrun --standalone --nproc_per_node=4 \
  scripts/prepare_hidden_states.py \
  --data-path $DATAPATH \
  --cache-dir $CACHE_DIR \
  --output-path $OUTPUT_PATH \
  --model $MODEL \
  --chat-template $CHAT_TEMPLATE \
  --tp $TP \
  --enable-aux-hidden-states \
  --aux-hidden-states-layers 1,17,33
#   --num-samples $NUM_SAMPLES