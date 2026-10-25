# greenmind-hackathon

Team 16 (Peacock): GreenMind AI Hackathon.
We serve `CMKL/MANGO1.5-Qwen3.5-9B` on one A100 40GB. Best graded run is **Final #19: S 99.8 / 298/300**. The goal is to answer as many requests on time as possible, using as little GPU energy as possible.

## Our approach: our own INT8 (W8A8) weights

**Problem we found.** At peak load our server spent most of its GPU time reading prompts (prefill): about 10,000 prompt tokens/s against about 360 generated tokens/s, with about 120 requests running. Generating tokens was starved, so many requests missed the 50 ms per-token limit.

**Why the prebuilt 4-bit AWQ model doesn't fix it.** W4A16 saves memory, but the weights are unpacked to 16-bit before each matrix multiply, so prompt reading runs at 16-bit speed.

**What we did.** We quantized the official weights to W8A8: 8-bit integer weights (one scale per output channel) and 8-bit integer activations (one scale per token, computed at run time). This runs the matrix multiplies on the A100's INT8 tensor cores, which have about twice the peak throughput of its 16-bit ones. vLLM confirms it uses `CutlassInt8ScaledMMLinearKernel` for these layers. On the best run we use MTP with 3 draft tokens (fixed, not adaptive) and a thinking-token budget middleware. That middleware only sets `thinking_token_budget`. It does not change max_tokens, temperature, or thinking on/off.

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

# 2. Power limit for the 99.8 run (held for the whole Final)
sudo nvidia-smi -pl 400

# 3. Serve (vLLM 0.32). Copy serving/thinking_budget.py to ~/ first.
bash serving/serve_best.sh               # port 8000, served name "mango"
```

## Results

Official grader, one A100 40GB:

| Setup | Track | Score | On time | P95 token | J/req | Accuracy |
|---|---|---|---|---|---|---|
| Prebuilt AWQ W4A16 (+ MTP) | Final | 0.0 | 90/300 | 113 ms | 195 | 87.1% |
| Our W8A8, mem 0.95, MTP 1, vLLM 0.27 | Final | 78.2 | 230/300 | 56 ms | 198 | 85.5% |
| W8A8 + vLLM 0.32, MTP 1, no cap | Final | 94.6 | 253/300 | 55 ms | 166 | 88.2% |
| + thinking cap + MTP 3, ~260 W | Final | 98.6 | 287/300 | 43 ms | 153 | 87.3% |
| **Same flags, 400 W, fresh process** | Final | **99.8** | **298/300** | **33 ms** | **172** | 86.2% |
| Same flags as 98.6 | Validation | 99.7 | 99/100 | 31 ms | 184 | 91.4% |

Full table: [`EXPERIMENTS.md`](EXPERIMENTS.md).

**99.8 vs 98.6.** Serve flags did not change. VM traces (`runlog/traces/20261025-074111.csv` vs `20261025-082700.csv`): at peak (≥30 running) the 98.6 run sat at **258 W / 1169 MHz**; the 99.8 run sat at **380 W / 1397 MHz**. The 99.8 process was also a clean start. Energy stayed full credit (3.55×; 98.6 was 3.84×).

**KV cache.** At mem 0.90 the INT8 KV cache (477k) hit 100% on Final. 0.95 fixed that (78.2). vLLM 0.32, thinking cap, and MTP K=3 are the later jumps.
