# Strata vs llama.cpp: two MoE expert caches on one old card

Both engines can now run a mixture-of-experts model that is bigger than the graphics card by keeping the busy experts in
VRAM and fetching the rest when needed. [Strata](https://github.com/Niko1221/Strata) was built around that idea;
llama.cpp added its own, experimental, in October 2026 (`--moe-cache-mib`, PR #29887, build b11480). This repo
compares them head to head on the same card, the same model file and the same small amount of RAM.

## The setup

| | |
|---|---|
| GPU | Tesla V100-PCIe 32 GB (sm_70, 2017), PCIe Gen3 x4 |
| CPU / RAM | i7-13700KF, **16 GB** RAM |
| Disk | NVMe (~2 GB/s) holding the model |
| Model | Qwen3.8-Flash-Next GSQ-RCO Coder IQ1_M (`qwen4exp`, 48 layers x 256 experts, 10 per token), 58.4 GB in two GGUF parts: the same files for both engines |
| Strata | 0.1.41 everyday config (resident RAM mode, `STRATA_SM70_TABLE=1`) |
| llama.cpp | b11718 (`547208dca`), built for sm_70 with CUDA 12.9 |

The model is far bigger than VRAM + RAM (32 + 16 GB), so whatever is not cached is read from the NVMe: the hard case
for an expert cache, and one llama.cpp's write-up did not test (its tests had the experts in host RAM).

## The plan

1. **Load check:** llama.cpp loads the model and gives sensible answers.
2. **Cache-size sweep (overnight):** llama.cpp decode and prompt speed, and cache hit rate, at several `--moe-cache-mib`
   sizes, fixed prompts, cold restarts.
3. **Reference:** Strata on the same prompts.
4. **Result:** a curve of llama.cpp speed against cache size, with Strata as the line to beat.

## Status

Started 2026-10-18. No results yet.
