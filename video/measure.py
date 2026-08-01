#!/usr/bin/env python3
# Measure single + aggregate throughput via llama.cpp native /completion (no model name needed).
import json, subprocess, sys, time, concurrent.futures as cf

PORT = sys.argv[1] if len(sys.argv) > 1 else "8082"
PROMPT = "Write a Python function to merge two sorted lists."
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def call():
    p = subprocess.run(
        ["curl", "-s", f"localhost:{PORT}/completion", "-H", "Content-Type: application/json",
         "-d", json.dumps({"prompt": PROMPT, "n_predict": 120})],
        capture_output=True, text=True)
    return json.loads(p.stdout)


# warm
try:
    call()
except Exception as e:
    print("warm failed:", e); sys.exit(1)

s = call()
t = s.get("timings", {})
print(f"SINGLE-STREAM: {t.get('predicted_per_second', 0):.1f} tok/s")

start = time.time()
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    res = list(ex.map(lambda _: call(), range(N)))
wall = time.time() - start
toks = sum(r.get("timings", {}).get("predicted_n", 0) for r in res)
print(f"AGGREGATE ({N} concurrent): {toks} tokens / {wall:.1f}s = {toks/wall:.1f} tok/s")
