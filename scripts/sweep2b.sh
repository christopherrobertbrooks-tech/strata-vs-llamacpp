#!/usr/bin/env bash
# Round 2b (Chris 2026-10-18: "just remove the image thing"): Strata's packs refuse to run without drafts (the native IQ
# pack needs --spec >= 2), so the like-for-like is Strata with its normal drafts and the image reader removed (all
# 12,288 experts on the card), twice; plus one try of the smallest window (--spec 2) without the draft helper.
# Waits for round 2 to finish first. Same client, fresh prompts.
set -u
R=/mnt/nvme/strata-vs-llamacpp; S=$R/repo/scripts; O=$R/repo/results/r2; L=$O/sweep.log; PY=/mnt/steam/Strata-0140/.venv/bin/python
say() { echo "$(date +%T) $*" | tee -a $L; }
until grep -q "ALL DONE" $L; do sleep 10; done
wait_up() { for i in $(seq 1 120); do curl -sf -m 3 localhost:$1/health > /dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
stop() { kill -TERM -- -$1 2>/dev/null; for i in $(seq 1 40); do kill -0 -- -$1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
strata() {   # strata <tag> <config>
  tag=$1; say "$tag: start"; cd /mnt/steam/Strata-0141
  setsid $PY serve/server.py --engine strata --config $2 --port 5842 > $O/$tag-server.log 2>&1 < /dev/null & pid=$!
  if ! wait_up 5842 $pid; then say "$tag: NOT UP ($(tail -2 $O/$tag-server.log | tr '\n' ' ' | cut -c1-300))"; stop $pid; return 1; fi
  say "$tag: up; V100 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1); $(grep -a -o -E 'expert cache [0-9]+ slots[^;]*' strata-coder-iq1_m.log | tail -1)"
  $PY $S/bench.py --fresh --url http://127.0.0.1:5842 --engine strata --label $tag --out $O/$tag.json >> $L 2>&1
  stop $pid; say "$tag: done"
}
say "ROUND 2b START"
strata strata-novision-p1 strata-coder-iq1_m-novision.json
strata strata-spec2-nomtp-p1 strata-coder-iq1_m-spec2-nomtp-novision.json
strata strata-novision-p2 strata-coder-iq1_m-novision.json
say "ROUND 2b DONE; V100 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
