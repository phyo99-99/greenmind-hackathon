"""
quant_w8a8.py — Team 16 (Peacock), GreenMind AI Hackathon

WHAT THIS DOES
  Makes our own 8-bit copy of the official CMKL/MANGO1.5-Qwen3.5-9B model.
  "W8A8" = Weights 8-bit AND Activations 8-bit (both stored as INT8 integers).

WHY WE ARE DOING IT
  Our server logs at peak load showed the GPU spending almost all its time
  READING PROMPTS ("prefill"): ~10,000 prompt tokens/s vs only ~360 generated
  tokens/s with 121 requests running. That means each request got ~3 tokens/s
  (~330 ms per token), far over the 50 ms limit.
  Reading prompts is limited by raw math speed (compute), not by memory.
  - The prebuilt AWQ model (W4A16) is 4-bit in memory, but it unpacks the
    weights back to 16-bit before multiplying, so the math is the same speed
    as the original model. It saves memory, not compute.
  - W8A8 does the math on the A100's INT8 tensor cores, which have about twice
    the peak speed of its 16-bit tensor cores. In theory prompt reading becomes
    much faster, leaving GPU time for generating tokens on time.
  This is a hypothesis until a Validation run measures it.

IS IT ALLOWED? (briefing slide 12)
  Yes: derived weights made during the event from the official weights.
  This method uses NO calibration data (weights are rounded to 8-bit directly;
  activations are converted to 8-bit on the fly while serving), so the
  "calibration data must be open" rule is trivially satisfied.

HOW TO RUN (on the VM, with no vLLM server running)
  nohup ~/lc-env/bin/python ~/quant_w8a8.py > ~/logs/quant_w8a8.log 2>&1 &
  tail -5 ~/logs/quant_w8a8.log          # check progress
  Output folder: ~/models/mango-w8a8-team16

MTP DRAFT HEAD
  llm-compressor automatically copies the MTP draft-head weights (used for
  speculative decoding) from the official checkpoint, unquantized. Log line:
  "save_mtp_tensors_to_checkpoint | INFO - Copied MTP weights ..."

RESULT OF FIRST RUN (Oct 25, 01:13)
  Took ~30 seconds. 248 Linear layers quantized to INT8.
"""
import os, glob, shutil
from transformers import AutoTokenizer
from llmcompressor import oneshot                       # llm-compressor: the tool CMKL also used
from llmcompressor.modifiers.quantization import QuantizationModifier

# Qwen3.5 is a vision+language model class; fall back to the generic class on older transformers.
try:
    from transformers import Qwen3_5ForConditionalGeneration as ModelCls
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelCls

# Locate the already-downloaded models in the Hugging Face cache.
hub = os.path.expanduser("~/.cache/huggingface/hub/")
M = glob.glob(hub + "models--CMKL--MANGO1.5-Qwen3.5-9B/snapshots/*/")[0]                # official BF16 weights (the source)
A = glob.glob(hub + "models--CMKL--MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp/snapshots/*/")[0]  # CMKL's AWQ build (only used for generation_config)
OUT = os.path.expanduser("~/models/mango-w8a8-team16")

tok = AutoTokenizer.from_pretrained(M)
model = ModelCls.from_pretrained(M, dtype="auto")        # loads on CPU (~18 GB RAM); GPU not needed

# The recipe:
#   targets="Linear"  -> quantize every matrix-multiply layer
#   scheme="W8A8"     -> INT8 weights (one scale per output channel) +
#                        INT8 activations computed per token at run time (no calibration)
#   ignore=[...]      -> keep these in 16-bit, same list CMKL used for their AWQ:
#                        lm_head (final word-choice layer, sensitive to rounding),
#                        visual (image encoder, unused: we serve --language-model-only),
#                        mtp (speculative-decoding draft head)
recipe = QuantizationModifier(targets="Linear", scheme="W8A8",
                              ignore=["lm_head", "re:.*visual.*", "re:.*mtp.*"])
oneshot(model=model, recipe=recipe)                      # no dataset= argument: data-free quantization

# Save in compressed-tensors format, which vLLM loads directly.
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)

# Copy the files vLLM needs for the chat template and processor, so prompts are
# formatted exactly like the official model (evaluation protocol unchanged).
for f in ["chat_template.jinja", "preprocessor_config.json", "processor_config.json"]:
    if os.path.exists(M + f): shutil.copy(M + f, OUT)
shutil.copy(A + "generation_config.json", OUT)
print("saved to", OUT)
