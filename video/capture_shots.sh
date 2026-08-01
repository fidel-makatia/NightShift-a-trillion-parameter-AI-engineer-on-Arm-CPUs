#!/usr/bin/env bash
# Capture real system state + a live ops task for screenshots. Runs on the Arm VM.
O=/data/shots; mkdir -p "$O"

# 1. system identity (the "no GPU" proof)
{
  echo '$ lscpu | grep -E "Architecture|Model name|CPU\(s\)|NUMA"'
  lscpu | grep -E "Architecture|Model name|^CPU\(s\)|NUMA node0 CPU|NUMA node1 CPU"
  echo
  echo '$ nvidia-smi'
  nvidia-smi 2>&1 || echo 'Command "nvidia-smi" not found  — there is no GPU on this machine.'
  echo
  echo '$ free -h | head -2'
  free -h | head -2
} > "$O/system.txt"

# 2. a real DevOps task: ask K2 for an ops one-liner while sampling core usage
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2","messages":[{"role":"user","content":"Give me a single bash one-liner to find the 10 largest files under /var/log, human-readable, newest first on ties. One line, no explanation."}],"max_tokens":160}' \
  > "$O/ops.json" &
REQ=$!
sleep 5
{
  echo '$ mpstat -P ALL 1 1   # while Kimi K2 generates on Arm'
  BUSY=$(mpstat -P ALL 1 1 2>/dev/null | awk '/Average/ && $2 ~ /^[0-9]+$/ {u=100-$NF; if(u>50) c++} END{print c+0}')
  echo "    cores >50% busy: ${BUSY} / 96   (token generation engages the whole socket)"
  echo
  echo '$ ps -o pcpu,rss,comm -p $(pgrep -f llama-server)'
  ps -o pcpu,rss,comm -p "$(pgrep -f llama-server | head -1)"
} > "$O/cores.txt" 2>/dev/null
wait $REQ

# 3. the actual answer K2 produced (a usable ops command)
python3 -c "import json;print(json.load(open('$O/ops.json'))['choices'][0]['message']['content'].strip()[:300])" > "$O/ops_answer.txt" 2>/dev/null

# 4. model load info from the server journal
sudo journalctl -u llama-server --no-pager 2>/dev/null | grep -iE "model type|model params|model size|n_expert|general.arch" | head -6 > "$O/modelinfo.txt"

echo "captured:"; ls -la "$O"
