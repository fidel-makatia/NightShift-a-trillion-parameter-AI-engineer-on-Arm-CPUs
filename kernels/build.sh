#!/usr/bin/env bash
# Build the quantized GEMV kernels for Neoverse-N2 (Cobalt 100). Run on the Arm VM.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# -mcpu=neoverse-n2 enables dotprod, sve2, i8mm, bf16 for this exact core.
cc -O3 -mcpu=neoverse-n2 -o "$HERE/gemv_q8" "$HERE/gemv_q8.c" -lm
echo "built $HERE/gemv_q8"
"$HERE/gemv_q8"
