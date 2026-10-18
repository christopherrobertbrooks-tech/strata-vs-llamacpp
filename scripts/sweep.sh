#!/usr/bin/env bash
# The cache-size sweep (Chris 2026-10-18: "go ahead, run the sweep"): llama.cpp b11718 with --cpu-moe and
# --moe-cache-mib 24000 (every expert fits) / 20000 / 16000 / 8000, and Strata 0.1.41 (everyday config) as the line
# to beat -- each a cold start on the V100 alone, the same client (bench.py), prompts 256/1000/4096, 3 runs each.
# 24 GB and Strata run twice (start and end) to show drift. Refuses if the V100 or llama-swap's Strata is in use.
set -u
R=/mnt/nvme/strata-vs-llamacpp; S=$R/repo/scripts; O=$R/repo/results; L=$O/sweep.log; mkdir -p $O
M=/mnt/nvme/Strata-data-0140/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
BIN=/mnt/nvme/llama.cpp-q2/build-sm70/bin; PY=/mnt/steam/Strata-0140/.venv/bin/python
say() { echo "$(date +%T) $*" | tee -a $L; }
[ -n "$(ss -ltnp | grep ':5803 ')" ] && { say "refusing: llama-swap's strata is loaded"; exit 1; }
[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -gt 500 ] && { say "refusing: V100 busy"; exit 1; }
wait_up() { for i in $(seq 1 120); do curl -sf -m 3 localhost:$1/health > /dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
stop() { kill -TERM -- -$1 2>/dev/null; for i in $(seq 1 40); do kill -0 -- -$1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
llama() {   # llama <cache MiB> <pass>
  tag=llama-$1-p$2; say "$tag: start"
  CUDA_VISIBLE_DEVICES=1 setsid $BIN/llama-server -m $M -ngl 99 --cpu-moe --moe-cache-mib $1 -c 8192 -fa on --jinja \
    --host 127.0.0.1 --port 5851 -lv 4 > $O/$tag-server.log 2>&1 < /dev/null & pid=$!
  if ! wait_up 5851 $pid; then say "$tag: NOT UP ($(grep -a -i -E 'error|out of memory|failed' $O/$tag-server.log | tail -2 | tr '\n' ' '))"; stop $pid; return 1; fi
  say "$tag: up; V100 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1); $(grep -a 'MoE cache size' $O/$tag-server.log | sed 's/.*impl: *//')"
  $PY $S/bench.py --url http://127.0.0.1:5851 --engine llama --label $tag --out $O/$tag.json >> $L 2>&1
  stop $pid
  say "$tag: done; $(grep -a 'llama_moe_cache: ubatch' $O/$tag-server.log | sed 's/.*llama_moe_cache: //' | tr '\n' ' ')"
}
strata() {   # strata <pass>
  tag=strata-0141-p$1; say "$tag: start"; cd /mnt/steam/Strata-0141
  setsid $PY serve/server.py --engine strata --config strata-coder-iq1_m.json --port 5842 > $O/$tag-server.log 2>&1 < /dev/null & pid=$!
  if ! wait_up 5842 $pid; then say "$tag: NOT UP"; stop $pid; return 1; fi
  say "$tag: up; V100 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
  $PY $S/bench.py --url http://127.0.0.1:5842 --engine strata --label $tag --out $O/$tag.json >> $L 2>&1
  stop $pid; say "$tag: done"
}
say "START"
llama 24000 1 || llama 23000 1
strata 1
llama 20000 1
llama 16000 1
llama 8000 1
strata 2
llama 24000 2 || llama 23000 2
say "ALL DONE; V100 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
