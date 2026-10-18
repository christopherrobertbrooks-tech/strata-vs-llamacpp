#!/usr/bin/env bash
# Step 1: llama.cpp b11718 loads Strata's Coder files on the V100 alone (CUDA_VISIBLE_DEVICES=1) with the experts left
# in host memory (--cpu-moe: mmap'd from the NVMe, the gateway has 16 GB RAM) and a 20 GB VRAM expert cache, and
# answers one coding prompt at temperature 0. Logs the cache's own lines and the speeds. Refuses if the V100 is busy.
set -u
M=/mnt/nvme/Strata-data-0140/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
B=/mnt/nvme/llama.cpp-q2/build-sm70/bin; O=/mnt/nvme/strata-vs-llamacpp; mkdir -p $O; L=$O/load-check.log
[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -gt 500 ] && { echo "refusing: V100 busy" | tee $L; exit 1; }
echo "$(date +%T) start; RAM free: $(free -g | awk '/Mem:/ {print $7}') GB" | tee $L
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "wall %es, peak RSS %MKB" timeout 900 $B/llama-completion -m $M -ngl 99 --cpu-moe \
  --moe-cache-mib 20000 -c 4096 -n 96 --temp 0 --seed 1 -no-cnv \
  -p "Write a short Python function that checks whether a string is a palindrome, ignoring case and spaces." >> $L 2>&1 < /dev/null
echo "$(date +%T) exit $?; RAM free: $(free -g | awk '/Mem:/ {print $7}') GB" | tee -a $L
