#!/usr/bin/env bash
# Prove 20+ tok/s on the interactive tier: Qwen3-30B-A3B on Arm, parallel + continuous batching.
set -uo pipefail
BIN=/opt/llama.cpp/build/bin/llama-server
MODEL=$(ls /data/models/qwen3-30b-a3b/*.gguf)
pkill -9 -f llama-server 2>/dev/null; sleep 3
nohup "$BIN" -m "$MODEL" --alias qwen3 --host 0.0.0.0 --port 8082 -c 16384 \
  --parallel 8 --cont-batching -t 64 --jinja --reasoning-budget 0 > /data/qwenserve.log 2>&1 &
for i in $(seq 1 60); do curl -sf localhost:8082/health >/dev/null 2>&1 && break; sleep 3; done

REQ='{"model":"qwen3","temperature":0.2,"max_tokens":120,"messages":[{"role":"user","content":"Write a Python function to merge two sorted lists."}]}'
curl -s localhost:8082/v1/chat/completions -H "Content-Type: application/json" -d "$REQ" >/dev/null 2>&1  # warm
S1=$(curl -s localhost:8082/v1/chat/completions -H "Content-Type: application/json" -d "$REQ")
echo "$S1" | python3 -c "import json,sys;t=json.load(sys.stdin)['timings'];print(f'SINGLE-STREAM: {t[\"predicted_per_second\"]:.1f} tok/s')"

mkdir -p /data/qagg3; rm -f /data/qagg3/*.json
START=$(python3 -c "import time;print(time.time())")
for j in $(seq 1 8); do
  curl -s localhost:8082/v1/chat/completions -H "Content-Type: application/json" -d "$REQ" > /data/qagg3/r$j.json &
done
wait
END=$(python3 -c "import time;print(time.time())")
python3 - "$START" "$END" <<'PY'
import json, glob, sys
wall = float(sys.argv[2]) - float(sys.argv[1])
toks = sum(json.load(open(f))["timings"]["predicted_n"] for f in glob.glob("/data/qagg3/r*.json"))
print(f"AGGREGATE (8 concurrent): {toks} tokens / {wall:.1f}s = {toks/wall:.1f} tok/s")
PY
