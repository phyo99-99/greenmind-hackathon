# Experiment Log: Team 16 (Peacock)

All numbers are from the official grader on one A100 40GB. Early rows are vLLM 0.27.1. From Final #15 we used vLLM 0.32.
Rules we followed: one change at a time; prompts, max_tokens, thinking mode, temperature and sampling never changed; one vLLM process; power limit held constant for each run.

## Final runs (300 requests, graded)

| # | Setup | Score | On time | P95 time per token | P95 first token | J/request | Energy vs baseline | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 1 | Early mango serve | 0.0 | 97/300 | 104 ms | 2.00 s | 220 | 0.90× | 86.9% |
| 2 | Official AWQ W4A16 + MTP | 0.0 | 91/300 | 76 ms | 4.99 s | 201 | 0.93× | 86.9% |
| 3 | Official AWQ W4A16 + MTP | 0.0 | 89/300 | 93 ms | 2.90 s | 231 | 0.79× | 86.0% |
| 4 | Early mango serve | 0.0 | 85/300 | 98 ms | 2.59 s | 222 | 0.79× | 87.9% |
| 5 | Earlier tuning attempt (prebuilt model) | 0.0 | 68/300 | 143 ms | 5.67 s | 193 | 0.72× | 84.6% |
| 6 | Earlier tuning attempt (prebuilt model) | 0.0 | 75/300 | 139 ms | 4.50 s | 189 | 0.81× | 88.7% |
| 7 | Crash | FAILED | — | — | — | — | — | — |
| 8 | Prebuilt AWQ W4A16 + MTP | 0.0 | 90/300 | 113 ms | 3.13 s | 195 | 0.95× | 87.1% |
| 9 | **Our W8A8** + MTP, GPU memory 0.90 | 50.1 | 174/300 | 59 ms | 1.75 s | 200 | 1.78× | 85.5% |
| 10 | Our W8A8 + MTP, GPU memory 0.95 | 78.2 | 230/300 | 56 ms | 1.38 s | 198 | 2.37× | 85.5% |
| 11 | + max-num-seqs 64 | 25.1 | 125/300 | 47 ms | 5.75 s | 190 | 1.34× | 87.8% |
| 12 | + max-num-seqs 96 | 73.8 | 211/300 | 60 ms | 1.41 s | 190 | 2.28× | 87.8% |
| 13 | seq-cap / queue (TTFT 4.62 s) | 22.3 | 119/300 | 48 ms | 4.62 s | 190 | 1.29× | 86.9% |
| 14 | vLLM 0.27 restart (≈ Final 10) | 78.0 | 229/300 | 57 ms | 1.39 s | 198 | 2.36× | 85.5% |
| 15 | vLLM 0.32, no thinking cap, MTP 1, prefill 1024 | 94.6 | 253/300 | 55 ms | 1.44 s | 166 | 3.12× | 88.2% |
| 16 | + thinking cap, MTP 1 | 94.0 | 259/300 | 55 ms | 1.57 s | 179 | 2.97× | 88.4% |
| 17 | thinking cap + prefill 2048 | 87.8 | 241/300 | 55 ms | 1.00 s | 178 | 2.78× | 89.0% |
| 18 | thinking cap + MTP 3 + prefill 1024, ~260 W | 98.6 | 287/300 | 43 ms | 1.59 s | 153 | 3.84× | 87.3% |
| **19** | **same flags, 400 W, fresh process** | **99.8** | **298/300** | **33 ms** | 1.75 s | 172 | **3.55×** | 86.2% |

## Validation runs (100 requests, practice)

| Setup | Score | On time | P95 time per token | Notes |
|---|---|---|---|---|
| Prebuilt BF16 (no quantization) | 22.2 | — | — | Reference point |
| Prebuilt AWQ + MTP, 0 / 1 / 2 / 3 draft tokens | 55.2 / **62.9** / 53.3 / 0.0 | — | — | 1 draft token was best. The same setup also scored 0.0 (27/100) on another run, so results were noisy |
| AWQ + MTP, cap of 32 / 48 / 64 requests at once | 0.0 | — | — | Too few slots, so requests waited too long for their first token |
| AWQ, `--max-num-batched-tokens 8192` (vs 4096) | worse | — | — | Bigger prompt chunks per step slowed token generation |
| AWQ, power limit 320 W (vs 260 W) | worse | — | — | More energy without enough extra on-time requests |
| AWQ, GPU memory 0.93 | 1.8 | 34/100 | 79 ms | |
| AWQ, no MTP, batched tokens 8192 | 0.0 | 38/100 | 83 ms | |
| AWQ, prefix caching (`align` mode) | 0.0 | 35/100 | 75 ms | Experimental for this hybrid model; only 1.4% of prompts reused |
| Our W8A8 + MTP, GPU memory 0.90 | 78.7 | 96/100 | 33 ms | Energy 2.62× baseline, accuracy 82.8% |
| W8A8 + vLLM 0.32 + thinking cap + MTP 3 (same as Final 18) | **99.7** | 99/100 | 31 ms | Practice only |
| Prefill 600 | 76.7 | 95/100 | 32 ms | Worse energy |
| PrefillGuard scheduler | 0.0 | 30/100 | 67 ms | Stalled the engine |
| DeadlineHide scheduler | 66.4 | 62/100 | 108 ms | Hid running decodes |

## Key findings

1. **The bottleneck was prompt reading (prefill), not memory.** At peak load the server read about 10,000 prompt tokens/s but generated only about 360 tokens/s across about 120 running requests, roughly 330 ms per token. Scheduler flags only moved this compute shortage around, which is why none of them helped.
2. **4-bit weights don't speed up the maths on an A100.** W4A16 unpacks weights to 16-bit before multiplying. Our W8A8 runs on the INT8 tensor cores instead. At a 260 W cap the 98.6 Final peak sat at **1169 MHz / 258 W**. At 400 W the 99.8 Final peak sat at **1397 MHz / 380 W** (card max 1410 MHz). See `traces/`.
3. **Bigger weights mean less KV cache.** INT8 weights left only 477k tokens of KV cache at memory 0.90. It hit 100% in the Final, with 7 preemptions. Raising memory to 0.95 (646k tokens) took the Final from 50.1 to 78.2.
4. **vLLM 0.32** (Final 15, 94.6) is a different engine than Final 14 (78.0 on 0.27). Not the same setup twice.
5. **Thinking cap + MTP 3** (Final 18, 98.6 / 287, P95 43 ms). Cap alone (Final 16) added a few on-time requests but lost energy. Adaptive K was not used.
6. **400 W + a fresh process** (Final 19, 99.8 / 298, P95 33 ms). Same serve flags as 18. Peak trace: 258 W / 1169 MHz vs 380 W / 1397 MHz. Energy still full marks.

## Failed or abandoned ideas

- **Re-quantizing AWQ ourselves:** abandoned. The script crashed in the processor step, and it would not have fixed the compute bottleneck anyway.
- **Prefix caching:** worse, as shown above.
- **320 W on the AWQ stack:** worse energy per on-time request. **400 W on the 98.6 stack** is the 99.8 row (peak no longer stuck at ~260 W).
- **max-num-seqs 64 / 96:** Validation looked fine; Final queued (25.1 / 73.8).
- **Prefill 600 and 2048:** worse energy or late tail on Final.
- **Our two hide/guard schedulers:** Validation 0.0 and 66.4. Code: `serving/prefill_guard.py`, `serving/deadline_hide.py`. Do not serve those.
