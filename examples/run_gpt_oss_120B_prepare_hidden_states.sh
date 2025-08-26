export DATAPATH="/root/.cache/user_artifacts/gpt-oss-120b/dataset/perfect-blend-gptoss-20B-1M.jsonl"
export CACHE_DIR="/root/.cache"
export OUTPUT_PATH="/root/data/yilian/data/gpt_oss_120B"
export MODEL="openai/gpt-oss-120B"
export CHAT_TEMPLATE="gpt-oss"
export TP=4

# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3

uv run -- \
torchrun --standalone --nproc_per_node=4 \
         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  scripts/prepare_hidden_states.py \
  --data-path "$DATAPATH" \
  --cache-dir "$CACHE_DIR" \
  --output-path "$OUTPUT_PATH" \
  --model "$MODEL" \
  --chat-template "$CHAT_TEMPLATE" \
  --tp "$TP" \
  --enable-aux-hidden-states \
  --dist-timeout 7200 \
  --aux-hidden-states-layers 1,17,33 \
  --start-idx 80000 \
  # --start-idx 702612