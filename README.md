# greenmind-hackathon

Team 16 (Peacock): GreenMind AI Hackathon.
We serve `CMKL/MANGO1.5-Qwen3.5-9B` on one A100 40GB with vLLM 0.27.1. The goal is to answer as many requests on time as possible, using as little GPU energy as possible.

## Our approach: our own INT8 (W8A8) weights

**Problem we found.** At peak load our server spent most of its GPU time reading prompts (prefill): about 10,000 prompt tokens/s against about 360 generated tokens/s, with about 120 requests running. Generating tokens was starved, so many requests missed the 50 ms per-token limit.

**Why the prebuilt 4-bit AWQ model doesn't fix it.** W4A16 saves memory, but the weights are unpacked to 16-bit before each matrix multiply, so prompt reading runs at 16-bit speed.

**What we did.** We quantized the official weights to W8A8: 8-bit integer weights (one scale per output channel) and 8-bit integer activations (one scale per token, computed at run time). This runs the matrix multiplies on the A100's INT8 tensor cores, which have about twice the peak throughput of its 16-bit ones. vLLM confirms it uses `CutlassInt8ScaledMMLinearKernel` for these layers. We also keep MTP speculative decoding (1 draft token).

## Weights statement

- **Source:** the official `CMKL/MANGO1.5-Qwen3.5-9B` BF16 weights. Derived during the event.
- **Method:** data-free W8A8 INT8 quantization with [llm-compressor](https://github.com/vllm-project/llm-compressor) (`QuantizationModifier`, scheme `W8A8`). See [`quantization/quant_w8a8.py`](quantization/quant_w8a8.py).
- **Calibration data:** none. Weights are rounded directly, and activation scales are computed per token at run time.
- **Kept in 16-bit:** `lm_head`, the vision encoder (unused), and the MTP draft head. This is the same ignore list as CMKL's own AWQ recipe.
- **No fine-tuning.** Prompts, chat template, max_tokens, thinking mode and sampling are unchanged (all set by the grader).

## How to reproduce

```bash
# 1. Quantize (about 30 s; runs on CPU)
python quantization/quant_w8a8.py        # writes ~/models/mango-w8a8-team16

# 2. Serve
bash serving/serve_w8a8.sh               # vLLM on port 8000, served name "mango"
```

The GPU power limit is set to 260 W and held constant for the whole run (`sudo nvidia-smi -pl 260`).

## Results

_Validation results to be added._
