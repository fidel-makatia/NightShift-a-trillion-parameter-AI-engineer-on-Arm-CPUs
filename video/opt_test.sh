#!/usr/bin/env bash
# Optimization harness: restart K3 server with a given thread count + extra flags,
# warm the cache, then time a generation. Usage: opt_test.sh <label> <threads> [extra llama args...]
set -uo pipefail
LABEL="$1"; THREADS="$2"; shift 2; EXTRA="$*"
BIN=/data/llama-k3/build/bin/llama-server
MODEL=$(ls /data/models/Kimi-K3-GGUF/UD-IQ1_S/*00001-of-*.gguf)
pkill -9 -f llama-server 2>/dev/null; sleep 3
nohup "$BIN" -m "$MODEL" --host 0.0.0.0 --port 8081 -c 4096 -t "$THREADS" \
  --jinja --reasoning-budget 0 $EXTRA > /data/k3serve.log 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:8081/health >/dev/null 2>&1 && break; sleep 5; done

# print the model metadata once (arch + expert counts)
grep -aiE "arch |n_expert|n_expert_used|expert_used_count" /data/k3serve.log | head -6

REQ='{"model":"kimi-k3","temperature":0.2,"max_tokens":80,"messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number iteratively."}]}'
# warm-up (pull weights into cache / prime)
curl -s localhost:8081/v1/chat/completions -H "Content-Type: application/json" -d "$REQ" >/dev/null 2>&1
# measured run
curl -s localhost:8081/v1/chat/completions -H "Content-Type: application/json" -d "$REQ" | \
  python3 -c "import json,sys;d=json.load(sys.stdin);t=d['timings'];print(f'RESULT [$LABEL] gen {t[\"predicted_per_second\"]:.2f} tok/s | prompt {t[\"prompt_per_second\"]:.2f} tok/s')"
