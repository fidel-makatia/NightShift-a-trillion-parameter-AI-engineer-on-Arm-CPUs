#!/usr/bin/env bash
# Clean single-shot K3 generation for the screenshot + honest timing.
BIN=/data/llama-k3/build/bin/llama-cli
MODEL=$(ls /data/models/Kimi-K3-GGUF/UD-IQ1_S/*00001-of-*.gguf)
rm -f /data/K3_CLEAN_DONE
"$BIN" -m "$MODEL" -t 64 -c 4096 -n 200 --no-warmup -no-cnv \
  -p "In two sentences, explain why a mixture-of-experts design lets a trillion-parameter model run on CPUs." \
  > /data/k3_clean_out.log 2>/data/k3_clean_err.log
echo "RC=$?" >> /data/k3_clean_err.log
touch /data/K3_CLEAN_DONE
