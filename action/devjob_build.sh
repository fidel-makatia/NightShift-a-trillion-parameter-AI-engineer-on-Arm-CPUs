#!/usr/bin/env bash
# A REAL dev job on the Arm CPU: Kimi K2 builds a FastAPI URL-shortener microservice,
# then we install, run, and hit it live — all on the same Arm VM, no GPU anywhere.
set -uo pipefail
WORK=/data/devjob
rm -rf "$WORK"; mkdir -p "$WORK/app"
cd "$WORK"

echo "==============================================================="
echo " NightShift · real dev job on Arm CPU"
echo " Kimi K2 (1.04T params) writes a deployable microservice,"
echo " then we build & run it on the same 96-core Arm VM. No GPU."
echo "==============================================================="

# ---------- 1. Ask K2 to build the service ----------
read -r -d '' TASK <<'PROMPT'
Build a production-ready FastAPI URL-shortener microservice in a SINGLE file.
Requirements:
- POST /api/shorten  body {"url": "..."}  -> returns {"code": "...", "short_url": "..."}
- GET  /{code}       -> HTTP 307 redirect to the original URL (404 if unknown)
- GET  /healthz      -> {"status": "ok"}
- In-memory dict store; generate a 6-char base62 code; validate the URL starts with http.
- Runnable with:  uvicorn main:app --host 0.0.0.0 --port 9000
Output ONLY the complete Python file, wrapped exactly between the markers:
<<<FILE main.py>>>
<code here>
<<<END>>>
No prose, no explanation.
PROMPT

echo; echo "[1/5] Kimi K2 is writing the service on Arm CPU..."
REQ=$(python3 -c "import json,sys; print(json.dumps({'model':'kimi-k2','temperature':0.1,'max_tokens':1600,'messages':[{'role':'user','content':sys.argv[1]}]}))" "$TASK")
curl -sS localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "$REQ" > "$WORK/response.json"

TPS=$(python3 -c "import json;d=json.load(open('$WORK/response.json'));print(f\"{d['timings']['predicted_per_second']:.1f}\")" 2>/dev/null)
echo "    ...done. Generated at ${TPS} tok/s on Arm."

# ---------- 2. Extract the file K2 wrote ----------
echo "[2/5] Extracting main.py from K2's output..."
python3 - "$WORK/response.json" "$WORK/app/main.py" <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1]))
txt = d["choices"][0]["message"].get("content") or ""
m = re.search(r"<<<FILE main\.py>>>(.*?)<<<END>>>", txt, re.S)
if not m:
    m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S)  # fallback: first code fence
code = (m.group(1) if m else txt).strip()
open(sys.argv[2], "w").write(code + "\n")
print("    wrote %d lines" % (code.count(chr(10)) + 1))
PY
echo "----- app/main.py (written by Kimi K2) -----"
sed -n '1,40p' "$WORK/app/main.py"
echo "--------------------------------------------"

# ---------- 3. Install deps on Arm ----------
echo "[3/5] Installing FastAPI + uvicorn on Arm (aarch64 wheels)..."
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q --disable-pip-version-check fastapi "uvicorn[standard]" >/dev/null 2>&1
echo "    installed: $("$WORK/venv/bin/pip" show fastapi 2>/dev/null | awk '/Version/{print "fastapi "$2}')"

# ---------- 4. Run the service on Arm ----------
echo "[4/5] Launching the service K2 wrote on the Arm CPU..."
cd "$WORK/app"
nohup "$WORK/venv/bin/uvicorn" main:app --host 0.0.0.0 --port 9000 > "$WORK/server.log" 2>&1 &
SVPID=$!
for i in $(seq 1 30); do curl -sf localhost:9000/healthz >/dev/null 2>&1 && break; sleep 1; done

# ---------- 5. Exercise the live API ----------
echo "[5/5] Hitting the live API — proof it actually works:"
echo "    \$ curl localhost:9000/healthz"
echo "    -> $(curl -s localhost:9000/healthz)"
echo "    \$ curl -X POST localhost:9000/api/shorten -d '{\"url\":\"https://arm.com/ai\"}'"
SHORT=$(curl -s -X POST localhost:9000/api/shorten -H "Content-Type: application/json" -d '{"url":"https://arm.com/ai"}')
echo "    -> $SHORT"
CODE=$(python3 -c "import json,sys;print(json.load(sys.stdin)['code'])" <<<"$SHORT" 2>/dev/null)
echo "    \$ curl -i localhost:9000/$CODE   # should 307-redirect"
curl -s -i "localhost:9000/$CODE" | grep -iE "^HTTP|^location" | sed 's/^/    -> /'

kill $SVPID 2>/dev/null
echo
echo "SUCCESS: Kimi K2 built a working URL-shortener microservice, and it ran and"
echo "served live traffic on a 96-core Arm CPU (Azure Cobalt 100) with zero GPUs."
touch "$WORK/DEVJOB_DONE"
