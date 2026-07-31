#!/usr/bin/env bash
# NightShift benchmark harness — runs ON the VM.
# Measures prompt-processing (pp) and token-generation (tg) rates across thread counts,
# with results appended as CSV for the playbook charts.
#
# Usage: ./bench.sh <model.gguf> [label]
set -euo pipefail

MODEL="${1:?usage: bench.sh <model.gguf> [label]}"
LABEL="${2:-$(basename "$MODEL" .gguf)}"
BIN=/opt/llama.cpp/build/bin/llama-bench
OUT="/data/bench/${LABEL}_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p /data/bench

echo "=== NightShift bench: $LABEL ==="
echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//'), $(nproc) cores"
echo "Output: $OUT"

# pp512 = prompt processing (compute-bound, where KleidiAI/i8mm shines)
# tg128 = token generation (memory-bandwidth-bound, MoE active-params story)
"$BIN" -m "$MODEL" \
  -p 512 -n 128 \
  -t 24,48,96 \
  --mmap 1 \
  -o csv | tee "$OUT"

echo "Done. Result: $OUT"
