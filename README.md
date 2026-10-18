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

## First sweep (2026-10-18)

Writing speed, tokens/s, median of the warm runs (runs 2-3; each run 1 is a cold start), 256 tokens out, drafts as each
engine ships them (Strata: MTP on, ~70% accepted; llama.cpp: none available for this model). Hit rate = llama.cpp's
own count for single-token steps.

| setting | cache hit rate | uploaded to the card | 256 | 1000 | 4096 |
|---|---|---|---|---|---|
| llama.cpp, 24 GB cache (all experts) | 99.2% | 17.7 GB | 38.2 | 39.2 | 38.5 |
| llama.cpp, 20 GB | 99.1% | 19.3 GB | 39.1 | 38.6 | 38.4 |
| llama.cpp, 16 GB | 96.4% | 78.7 GB | 23.0 | 22.7 | 25.0 |
| llama.cpp, 8 GB | 82.8% | 374.7 GB | 5.3 | 5.6 | 5.2 |
| **Strata 0.1.41** (everyday) | ~95% in VRAM, rest in locked RAM | - | **95.8** | **88.2** | **81.8** |

- Flat down to 20 GB, a cliff between 20 and 16 GB (3 points of hit rate = 4x the traffic over a PCIe Gen3 x4 slot),
  collapse at 8 GB.
- Even with every expert on the card llama.cpp writes at ~39 tok/s, about half of Strata. Its decode graph has 50
  splits per token (one per layer: the cache maps the router's picks on the CPU).
- Not yet fair: Strata drafts and llama.cpp can't here; prompt reading isn't measured right (the same prompt three
  times, and the longer prompts share their start: both engines reuse it).

Next: drafts off for both, llama.cpp with everything on the GPU and no cache (the control), fresh prompts.
