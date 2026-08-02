#!/usr/bin/env bash
# NightShift — clone and run, no hassle.
#   git clone <repo> && cd NightShift-* && ./run.sh
#
# What this does WITHOUT any cloud account, model download, or setup:
#   1. builds + runs the hand-written Arm GEMM kernel benchmark (self-contained, ~5s)
#   2. reproduces the ExpertAtlas expert-routing analysis from committed real data
#   3. regenerates the benchmark charts
# The full trillion-parameter stack (K2/K3 on Azure Cobalt) needs a VM — see infra/.
set -uo pipefail
cd "$(dirname "$0")"
arch=$(uname -m)
echo "=============================================="
echo " NightShift — clone-and-run  (arch: $arch)"
echo "=============================================="

echo
echo "[1/3] Arm GEMM kernel benchmark (no dependencies)"
if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then
  if cc -O3 -mcpu=native -o /tmp/ns_gemm kernels/gemm_q8.c -lm 2>/tmp/ns_build.log \
     || cc -O3 -o /tmp/ns_gemm kernels/gemm_q8.c -lm 2>/tmp/ns_build.log; then
    /tmp/ns_gemm
  else
    echo "  build failed (see /tmp/ns_build.log); measured numbers are in kernels/results.txt"
  fi
else
  echo "  You're on $arch — the kernel needs an Arm CPU. Measured results: kernels/results.txt"
fi

echo
echo "[2/3] ExpertAtlas — routing-skew analysis from committed data"
if command -v python3 >/dev/null 2>&1 && python3 -c "import matplotlib,numpy" 2>/dev/null; then
  python3 autotuner/analyze.py autotuner/results/k2_activations.csv --outdir /tmp/ns_atlas 2>/dev/null \
    && echo "  -> charts + placement policy in /tmp/ns_atlas/"
  echo
  echo "[3/3] Regenerating benchmark charts"
  python3 bench/plot_pro.py >/dev/null 2>&1 && echo "  -> playbook/artifacts/pro_*.png"
else
  echo "  (install deps to reproduce charts:  pip install matplotlib numpy)"
  echo "[3/3] skipped (needs matplotlib)"
fi

echo
echo "Done. Full deploy on Azure Cobalt:  cd infra && terraform apply   (see README)"
