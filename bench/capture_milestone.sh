#!/usr/bin/env bash
# Captures the "1T params on Arm CPUs" milestone artifacts — runs ON the VM.
# Everything lands in /data/artifacts/ for the playbook and Devpost gallery.
set -uo pipefail
OUT=/data/artifacts
mkdir -p $OUT

echo "=== 1. System identity (the 'no GPU on Earth' proof) ==="
{
  echo "# NightShift host — captured $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "## CPU"
  lscpu | grep -E "Architecture|Model name|CPU\(s\)|Vendor"
  echo "## Memory"
  free -h | head -2
  echo "## GPU (spoiler: none)"
  lspci 2>/dev/null | grep -i -E "vga|nvidia|3d" || echo "No GPU present."
  echo "## Azure VM"
  curl -s -H Metadata:true "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01" \
    | jq -r '"\(.vmSize) in \(.location), zone \(.zone)"'
  echo "## Model on disk"
  du -sh /data/models/Kimi-K2-Thinking-GGUF/UD-Q2_K_XL 2>/dev/null
} | tee $OUT/sysinfo.txt

echo; echo "=== 2. Timed generation (native endpoint exposes timings) ==="
curl -sS localhost:8080/completion -H "Content-Type: application/json" -d '{
  "prompt": "The three most important considerations when optimizing LLM inference for Arm CPUs are",
  "n_predict": 200
}' > $OUT/timed_completion.json
jq '{tokens_generated: .timings.predicted_n,
     generation_tok_per_sec: .timings.predicted_per_second,
     prompt_tokens: .timings.prompt_n,
     prompt_tok_per_sec: .timings.prompt_per_second,
     ttft_ms: .timings.prompt_ms}' $OUT/timed_completion.json | tee $OUT/timings_summary.json

echo; echo "=== 3. Chat sample (for the demo reel) ==="
curl -sS localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "kimi-k2",
  "messages": [{"role": "user", "content": "In two sentences: you are a 1-trillion-parameter model running on 96 Arm CPU cores with no GPU. How is that possible?"}],
  "max_tokens": 300
}' > $OUT/chat_sample.json
jq -r '.choices[0].message.content' $OUT/chat_sample.json

echo; echo "=== 4. Memory pressure under load ==="
free -h | tee $OUT/memory_under_load.txt

echo; echo "All artifacts in $OUT:"; ls -la $OUT
