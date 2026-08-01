#!/usr/bin/env bash
# Kimi K3 (2.8T, on Arm CPU) generates a complete responsive web app. Runs on the VM.
set -uo pipefail
O=/data/webapp; mkdir -p "$O"
PORT=8081

# wait for the K3 server
for i in $(seq 1 90); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break; sleep 10; done

read -r -d '' TASK <<'PROMPT'
Create a COMPLETE single-file web app in one index.html (inline CSS and vanilla JS, no external
libraries or CDNs). Build a polished, responsive "Cloud Cost Dashboard" for a SaaS:
- Dark theme, modern typography, subtle gradients, rounded cards, good spacing.
- A left sidebar with the product name "NightShift" and nav items (Overview, Usage, Billing, Settings).
- A top row of 4 KPI cards: Monthly spend, Active VMs, Tokens served, Avg cost / 1M tokens.
- A bar chart drawn with pure CSS/divs showing spend across 7 days.
- A recent-activity table with 5 rows (time, service, region, status badge).
- Must be mobile-responsive: sidebar collapses on narrow screens.
Output ONLY the file, wrapped exactly between:
<<<HTML>>>
...full index.html...
<<<END>>>
No prose.
PROMPT

echo "Kimi K3 is writing the web app on Arm CPU (this is slow at ~2 tok/s)..."
REQ=$(python3 -c "import json,sys;print(json.dumps({'model':'kimi-k3','temperature':0.3,'max_tokens':4000,'messages':[{'role':'user','content':sys.argv[1]}]}))" "$TASK")
curl -sS localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "$REQ" > "$O/resp.json"
python3 - "$O/resp.json" "$O/index.html" <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1]))
m = d["choices"][0]["message"]
txt = (m.get("content") or "") + "\n" + (m.get("reasoning_content") or "")
mt = re.search(r"<<<HTML>>>(.*?)<<<END>>>", txt, re.S) or re.search(r"```(?:html)?\s*(.*?)```", txt, re.S)
html = (mt.group(1) if mt else txt).strip()
open(sys.argv[2], "w").write(html)
try:
    tps = d["timings"]["predicted_per_second"]; n = d["timings"]["predicted_n"]
    print(f"generated {n} tokens at {tps:.2f} tok/s -> {len(html)} bytes html")
except Exception:
    print(f"wrote {len(html)} bytes html")
PY
echo "DONE"; touch "$O/WEBAPP_DONE"
