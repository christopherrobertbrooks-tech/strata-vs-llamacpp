#!/usr/bin/env bash
# Round 2 (Chris 2026-10-18: "run round 2"): make it fair. Fresh prompts every run (bench.py --fresh), and:
#   llama-gpu  : llama.cpp with the whole model on the V100 the normal way, no cache (the control; the 27 GB
#                per-layer lookup table stays in host memory, as llama.cpp keeps it with the cache too)
#   llama-24000: llama.cpp with the 24 GB cache (every expert on the card)
#   strata-nn  : Strata with its drafts off AND the image reader off (all 12,288 experts on the card) -- the like-for-like
#   strata     : Strata everyday (drafts on, image reader on, 95% of experts on the card)
# Each a cold start on the V100 alone; llama-gpu and strata-nn twice (start and end). Refuses if the V100 is in use.
set -u
R=/mnt/nvme/strata-vs-llamacpp; S=$R/repo/scripts; O=$R/repo/results/r2; L=$O/sweep.log; mkdir -p $O
M=/mnt/nvme/Strata-data-0140/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
BIN=/mnt/nvme/llama.cpp-q2/build-sm70/bin; PY=/mnt/steam/Strata-0140/.venv/bin/python
say() { echo "$(date +%T) $*" | tee -a $L; }
[ -n "$(ss -ltnp | grep ':5803 ')" ] && { say "refusing: llama-swap's strata is loaded"; exit 1; }
[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -gt 500 ] && { say "refusing: V100 busy"; exit 1; }
wait_up() { for i in $(seq 1 120); do curl -sf -m 3 localhost:$1/health > /dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
stop() { kill -TERM -- -$1 2>/dev/null; for i in $(seq 1 40); do kill -0 -- -$1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
llama() {   # llama <gpu|24000> <pass>
  tag=llama-$1-p$2; say "$tag: start"
  if [ $1 = gpu ]; then X=(-ot "per_layer_token_embd=CPU"); else X=(--cpu-moe --moe-cache-mib $1); fi
  CUDA_VISIBLE_DEVICES=1 setsid $BIN/llama-server -m $M -ngl 99 "${X[@]}" -c 8192 -fa on --jinja \
    --host 127.0.0.1 --port 5851 -lv 4 > $O/$tag-server.log 2>&1 < /dev/null & pid=$!
  if ! wait_up 5851 $pid; then say "$tag: NOT UP ($(grep -a -i -E 'error|out of memory|failed' $O/$tag-server.log | tail -2 | tr '\n' ' '))"; stop $pid; return 1; fi
  say "$tag: up; V100 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1); $(grep -a -E 'MoE cache size|splits =' $O/$tag-server.log | sed 's/.*: *//' | tr '\n' ' ')"
  $PY $S/bench.py --fresh --url http://127.0.0.1:5851 --engine llama --label $tag --out $O/$tag.json >> $L 2>&1
  stop $pid; say "$tag: done"
}
strata() {   # strata <nn|day> <pass>
  tag=strata-$1-p$2; say "$tag: start"; cd /mnt/steam/Strata-0141
  C=strata-coder-iq1_m.json; [ $1 = nn ] && C=strata-coder-iq1_m-nodraft-novision.json
  setsid $PY serve/server.py --engine strata --config $C --port 5842 > $O/$tag-server.log 2>&1 < /dev/null & pid=$!
  if ! wait_up 5842 $pid; then say "$tag: NOT UP ($(tail -3 $O/$tag-server.log | tr '\n' ' '))"; stop $pid; return 1; fi
  say "$tag: up; V100 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1); $(grep -a -o -E 'expert cache [0-9]+ slots[^;]*' strata-coder-iq1_m.log | tail -1)"
  $PY $S/bench.py --fresh --url http://127.0.0.1:5842 --engine strata --label $tag --out $O/$tag.json >> $L 2>&1
  stop $pid; say "$tag: done"
}
say "START"
llama gpu 1
strata nn 1
llama 24000 1
strata day 1
strata nn 2
llama gpu 2
say "ALL DONE; V100 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
