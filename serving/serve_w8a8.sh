#!/usr/bin/env bash
# serve_w8a8.sh — Team 16 (Peacock): serve our own W8A8 (INT8) model.
#
# Exactly the same flags as our best run so far (62.9 points, prebuilt AWQ + MTP 1),
# with ONE change: the model path points to our W8A8 build (see quant_w8a8.py).
# That keeps the Validation comparison fair: one change at a time.
#
# Run on the VM, inside the vLLM environment, with no other vLLM server running.

source "$HOME/vllm-env/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /home/researcher/models/mango-w8a8-team16 \
  --served-model-name mango \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --host 0.0.0.0 --port 8000 --language-model-only --max-model-len 49152 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gdn-prefill-backend triton --no-enable-flashinfer-autotune 2>&1 | tee ~/logs/w8a8.log
