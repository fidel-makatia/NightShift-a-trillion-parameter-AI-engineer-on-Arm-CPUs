#!/usr/bin/env bash
# Build ExpertAtlas's expert-capture tool against the existing llama.cpp build.
# Run on the VM (or anywhere llama.cpp is built). Expects LLAMA_DIR to point at a
# llama.cpp checkout that has been built (headers + libs present).
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_DIR/build}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# llama.cpp ships shared libs under build/bin. Link against them and the public headers.
c++ -std=c++17 -O2 "$HERE/expert-capture.cpp" \
  -I"$LLAMA_DIR/include" -I"$LLAMA_DIR/ggml/include" \
  -L"$BUILD_DIR/bin" \
  -lllama -lggml -lggml-base \
  -Wl,-rpath,"$BUILD_DIR/bin" \
  -o "$HERE/expert-capture"

echo "built $HERE/expert-capture"
