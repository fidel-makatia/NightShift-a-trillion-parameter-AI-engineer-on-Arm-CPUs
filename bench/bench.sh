#!/usr/bin/env bash
# NightShift benchmark harness — runs ON the VM.
# Measures prompt-processing (pp) and token-generation (tg) rates across thread counts,
# with results as CSV for the playbook charts.
#
# Usage: ./bench.sh <model.gguf> [label]
#
# Hardened per NightShift's own review of PR #7 (see playbook/README.md, Finding 4):
# validated inputs, safe glob handling, and the inference server is always restarted on exit.
set -euo pipefail

MODEL="${1:?usage: bench.sh <model.gguf> [label]}"
LABEL="${2:-$(basename "$MODEL" .gguf)}"
BIN="${LLAMA_BENCH:-/opt/llama.cpp/build/bin/llama-bench}"
BENCH_DIR="${BENCH_DIR:-/data/bench}"
THREADS="${THREADS:-24,48,96}"

[ -e "$MODEL" ] || { echo "error: model not found: $MODEL" >&2; exit 1; }
[ -x "$BIN" ]   || { echo "error: llama-bench not executable: $BIN" >&2; exit 1; }
mkdir -p "$BENCH_DIR"

# Always bring the inference server back, even if the benchmark aborts.
restore_server() { systemctl is-enabled --quiet llama-server 2>/dev/null && sudo systemctl start llama-server || true; }
trap restore_server EXIT

sudo systemctl stop llama-server 2>/dev/null || true

OUT="$BENCH_DIR/${LABEL}_$(date +%Y%m%d_%H%M%S).csv"
echo "=== NightShift bench: $LABEL ==="
echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//'), $(nproc) cores"
echo "Output: $OUT"

# pp512 = prompt processing (compute-bound, where KleidiAI/i8mm shines)
# tg128 = token generation (memory-bandwidth-bound, MoE active-params story)
"$BIN" -m "$MODEL" -p 512 -n 128 -t "$THREADS" --load-mode mmap -o csv | tee "$OUT"

echo "Done. Result: $OUT"
