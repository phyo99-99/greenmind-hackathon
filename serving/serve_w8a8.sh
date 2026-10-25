#!/usr/bin/env bash
# serve_w8a8.sh — historical Final #10 (78.2). Not the 99.8 command.
# Current best: serving/serve_best.sh (vLLM 0.32, thinking cap, MTP 3, 400 W).

source "$HOME/vllm-env/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /home/researcher/models/mango-w8a8-team16 \
  --served-model-name mango \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --host 0.0.0.0 --port 8000 --language-model-only --max-model-len 49152 \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gdn-prefill-backend triton --no-enable-flashinfer-autotune 2>&1 | tee ~/logs/w8a8_095.log
