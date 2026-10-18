# Strata vs llama.cpp: two MoE expert caches on one old card

Both engines can now run a mixture-of-experts model that is bigger than the graphics card by keeping the busy experts in
VRAM and fetching the rest when needed. [Strata](https://github.com/Niko1221/Strata) was built around that idea;
llama.cpp added its own, experimental, in October 2026 (`--moe-cache-mib`, PR #29887, build b11480). This repo
compares them head to head on the same card, the same model file and the same small amount of RAM.

**Result, in one line:** on this model and this card, Strata writes about **2.4x** faster and reads prompts about
**2.9x** faster than llama.cpp at its best -- and still about 1.5x faster at writing with its draft helper removed. This
is one model (Strata's home model) on one 2017 card; llama.cpp runs far more models and hardware, and on other models
it can be the faster one (e.g. [Strata #2464](https://github.com/Niko1221/Strata/issues/2464), Qwen3.6 at long context).

## The setup

| | |
|---|---|
| GPU | Tesla V100-PCIe 32 GB (sm_70, 2017), PCIe Gen3 x4 (3.3 GB/s measured) |
| CPU / RAM | i7-13700KF, **16 GB** RAM |
| Disk | NVMe (~2 GB/s) holding the model |
| Model | Qwen3.8-Flash-Next GSQ-RCO Coder IQ1_M (`qwen4exp`, 48 layers x 256 experts, 10 per token): the same two GGUF files for both engines (54.4 GiB: 23.4 GiB of experts, a 26.8 GiB per-layer lookup table that reads only a few rows per token, ~4 GiB core) |
| Strata | 0.1.41, everyday config (resident RAM mode, `STRATA_SM70_TABLE=1`, MTP drafts `--spec 4`) |
| llama.cpp | b11718 (`547208dca`), built for sm_70 with CUDA 12.9, `-fa on`, `-c 8192` |

## Method

- One client for both ([scripts/bench.py](scripts/bench.py)): OpenAI chat requests, streamed, temperature 0,
  presence_penalty 0, thinking off, 256 tokens out; prompts of 256 / 1000 / 4096 tokens made with the model's own
  tokenizer. Timed on the client: first token (prompt reading) and first-to-last token (writing).
- Round 2 (the numbers below): every request starts with its own line ("Session 4096-2") so no engine can reuse an
  earlier prompt, plus a 1000-token warm-up. Each setup is a cold start on the V100 alone; 3 runs per size; the
  llama.cpp control and Strata without the image reader ran twice, and the repeats agree within 1%.

## Results (round 2, fresh prompts)

Median tokens/s writing, and seconds to the first token:

| setup | experts on the card | write 256 | write 1000 | write 4096 | first token, 4096-token prompt |
|---|---|---|---|---|---|
| llama.cpp, whole model on the GPU the normal way (no cache) | 100% | 42.9 | 42.4 | 41.7 | 9.8 s (~420 tok/s) |
| llama.cpp, `--cpu-moe --moe-cache-mib 24000` | 100% (cache) | 35.2 | 37.5 | 37.3 | 13.2 s (~310 tok/s) |
| Strata, image reader removed, `--spec 2` and no MTP drafter | 100% | 70.9 | 66.7 | 60.4 | 3.4 s (~1,200 tok/s) |
| Strata, everyday (image reader on the GPU) | 95% + 5% in locked RAM | 92.4 | 88.5 | 84.8 | 3.4 s (~1,190 tok/s) |
| **Strata, image reader removed** | 100% | **101.2** | **98.5** | **98.0** | **3.4 s (~1,210 tok/s)** |

- **Like for like** (every expert on the card, each engine's normal settings): Strata ~100 vs llama.cpp ~42 writing
  (2.4x), ~1,200 vs ~420 tok/s reading (2.9x).
- **Not just the drafts:** Strata's IQ pack refuses `--spec` below 2, but with the MTP drafter removed it still writes
  61-71 tok/s, ~1.5x llama.cpp (which has no drafter for this model: the publisher ships no llama.cpp MTP head).
- **llama.cpp's cache costs ~10%** writing vs loading the same experts normally: its decode graph has 50 splits per
  token (the cache maps each layer's router picks on the CPU) against 2 without it.

## The cache-size sweep (round 1)

llama.cpp with less cache than experts. Writing tok/s, warm runs (round 1 reused prompts, so its prompt times are not
shown); hit rate and traffic are llama.cpp's own counters for single-token steps.

| `--moe-cache-mib` | share of experts | hit rate | copied to the card | write 256 | 1000 | 4096 |
|---|---|---|---|---|---|---|
| 24000 | 100% | 99.2% | 17.7 GB | 38.2 | 39.2 | 38.5 |
| 20000 | 83% | 99.1% | 19.3 GB | 39.1 | 38.6 | 38.4 |
| 16000 | 67% | 96.4% | 78.7 GB | 23.0 | 22.7 | 25.0 |
| 8000 | 33% | 82.8% | 374.7 GB | 5.3 | 5.6 | 5.2 |

Flat down to 83% of the experts, then a cliff: three points of hit rate quadruple the traffic over a Gen3 x4 slot,
and at a third of the experts the run is waiting on the NVMe most of the time.

## Files

- [scripts/](scripts): the client, both sweeps, the load check.
- [results/2026-10-18-sweep1](results/2026-10-18-sweep1), [results/2026-10-18-round2](results/2026-10-18-round2): every
  run (JSON with the replies), the sweep logs and each engine's own cache / split lines.

Measured 2026-10-18 by christopherrobertbrooks-tech.
