#!/usr/bin/env bash
# Final #19 — S 99.8 / 298/300
# vLLM 0.32, W8A8, thinking cap, MTP 3, prefill 1024
# Hold 400 W for the whole run: sudo nvidia-smi -pl 400
set -euo pipefail
source "$HOME/vllm032-env/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$HOME${PYTHONPATH:+:$PYTHONPATH}"

vllm serve /home/researcher/models/mango-w8a8-team16 \
  --served-model-name mango \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 0.0.0.0 --port 8000 --language-model-only --max-model-len 49152 \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 1024 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gdn-prefill-backend triton --no-enable-flashinfer-autotune \
  --middleware thinking_budget.ThinkingBudgetMiddleware
