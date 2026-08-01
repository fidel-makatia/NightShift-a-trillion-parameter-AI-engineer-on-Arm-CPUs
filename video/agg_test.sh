#!/usr/bin/env bash
# Measure AGGREGATE throughput: N concurrent requests through continuous batching.
# This is the real "shared dev server" metric. Usage: agg_test.sh <n_parallel> <n_concurrent>
set -uo pipefail
NP="${1:-8}"; NC="${2:-8}"
BIN=/data/llama-k3/build/bin/llama-server
MODEL=$(ls /data/models/Kimi-K3-GGUF/UD-IQ1_S/*00001-of-*.gguf)
pkill -9 -f llama-server 2>/dev/null; sleep 3
nohup "$BIN" -m "$MODEL" --host 0.0.0.0 --port 8081 -c $((NP*2048)) -t 64 \
  --parallel "$NP" --cont-batching --jinja --reasoning-budget 0 > /data/k3serve.log 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:8081/health >/dev/null 2>&1 && break; sleep 5; done

echo "cache/mem:"; free -g | awk 'NR<=2'
mkdir -p /data/agg; rm -f /data/agg/*.json
REQ='{"model":"kimi-k3","temperature":0.2,"max_tokens":80,"messages":[{"role":"user","content":"Write a short Python function to check if a string is a palindrome."}]}'
# warm one
curl -s localhost:8081/v1/chat/completions -H "Content-Type: application/json" -d "$REQ" >/dev/null 2>&1

START=$(date +%s.%N)
for j in $(seq 1 "$NC"); do
  curl -s localhost:8081/v1/chat/completions -H "Content-Type: application/json" -d "$REQ" > /data/agg/r$j.json &
done
wait
END=$(date +%s.%N)

python3 - "$START" "$END" "$NC" <<'PY'
import json, glob, sys
start, end, nc = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
wall = end - start
toks = 0; per = []
for f in glob.glob("/data/agg/r*.json"):
    try:
        d = json.load(open(f)); t = d["timings"]
        toks += t["predicted_n"]; per.append(t["predicted_per_second"])
    except Exception: pass
print(f"RESULT concurrent={nc}  wall={wall:.1f}s  total_gen_tokens={toks}")
print(f"AGGREGATE throughput = {toks/wall:.2f} tok/s   (per-stream avg {sum(per)/len(per):.2f} tok/s)")
PY
grep -aiE "expert_used_count|n_expert_used" /data/k3serve.log | head -2
