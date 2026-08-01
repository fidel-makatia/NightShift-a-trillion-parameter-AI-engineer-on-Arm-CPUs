#!/usr/bin/env bash
# Four concurrent dev jobs hitting ONE Kimi K3 endpoint on the Arm VM, served in parallel
# via continuous batching. Demonstrates a shared team inference server on CPU.
set -uo pipefail
O=/data/jobs; mkdir -p "$O"
PORT=8081
for i in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break; sleep 10; done

fire() {  # name  max_tokens  prompt
  local name="$1" maxtok="$2" prompt="$3"
  local req
  req=$(python3 -c "import json,sys;print(json.dumps({'model':'kimi-k3','temperature':0.2,'max_tokens':int(sys.argv[2]),'messages':[{'role':'user','content':sys.argv[1]}]}))" "$prompt" "$maxtok")
  curl -sS localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "$req" > "$O/$name.json"
  echo "$name done"
}

echo "firing 4 concurrent jobs at the K3 endpoint..."
fire codegen 600 "Write a thread-safe, O(1) LRU cache class in Python (get/put) using a dict plus a doubly linked list. Include a short docstring. Code only." &
fire webapp 3000 "Create a COMPLETE single index.html (inline CSS+JS, no CDNs): a responsive dark-theme 'Cloud Cost Dashboard' with a sidebar (NightShift + nav), 4 KPI cards, a pure-CSS 7-day bar chart, and a 5-row activity table with status badges. Sidebar collapses on mobile. Output only the HTML." &
fire embedded 700 "Write bare-metal C for an Arm Cortex-M4 (STM32) that blinks an LED on GPIO PA5 using direct CMSIS register access (RCC, GPIOA MODER/ODR) and a SysTick-based millisecond delay. No HAL. Code only, with brief comments." &
fire devtest 700 "Write a pytest test suite for 'def parse_iso8601(s: str) -> datetime'. Cover: valid UTC 'Z', valid offset '+02:00', naive datetime, invalid string (raises ValueError), and empty input. Code only." &
wait
echo "ALL DONE"
# summarize timings
python3 - "$O" <<'PY'
import json, glob, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    name = os.path.basename(f)[:-5]
    try:
        d = json.load(open(f)); t = d["timings"]
        c = d["choices"][0]["message"]
        body = (c.get("content") or c.get("reasoning_content") or "")
        print(f"{name:9s} {t['predicted_n']:4d} tok  {t['predicted_per_second']:5.2f} tok/s  {len(body)} chars")
    except Exception as e:
        print(f"{name}: parse error {e}")
PY
touch "$O/JOBS_DONE"
