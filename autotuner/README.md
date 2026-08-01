# ExpertAtlas

**Profile which experts a MoE model actually uses on your workload, then fit the model in
less RAM by keeping only the hot experts resident.**

Kimi K2 has 384 experts per layer and routes each token through just 8. If activation is
skewed — and the literature says MoE routing usually is ([~15–20% of experts serve ~80% of
tokens](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)) — then most
of the 360 GB of weights are cold most of the time. ExpertAtlas measures that skew **on real
developer workloads** (PR diffs, bug-fix sessions, code chat) and turns it into a concrete
memory-placement policy: hot experts pinned in RAM, cold experts memory-mapped from NVMe.

The payoff: run the same trillion-parameter model on a **smaller, cheaper Arm VM** — turning
the playbook's "you need 672 GiB" into "you need far less, if you place experts by usage."

## How it works

1. **Capture** (`expert-capture.cpp`) — a ~150-line tool that hooks llama.cpp's graph
   eval callback and records the router's `ffn_moe_topk` tensor (the selected expert IDs)
   for every token, every layer, over a workload. Output: `activations.csv` (layer, expert, count).
2. **Analyze** (`analyze.py`) — computes the activation skew (Gini + a Lorenz-style cumulative
   curve), renders a layer×expert heat map, and emits a **placement policy** for a chosen RAM
   budget: which `(layer, expert)` cells stay resident.
3. **Apply** — the policy becomes llama.cpp `--override-tensor` arguments that pin cold expert
   FFN tensors (`blk.<il>.ffn_*_exps`) to a disk-backed buffer, freeing RAM.

## Run it

```bash
# on the VM, against the built llama.cpp
LLAMA_DIR=/opt/llama.cpp ./build.sh
./expert-capture -m /data/models/.../Kimi-K2-*00001-of-*.gguf \
    -f workloads/dev.txt -o activations.csv -t 64

python3 analyze.py activations.csv --keep-frac 0.5 --outdir out/
# → out/skew_curve.png, out/expert_heatmap.png, out/placement.json, out/override-tensor.txt
```

## Status

- `analyze.py` — **done and self-tested** offline (synthetic fixture; the pipeline computes
  skew, curve, heat map, and placement correctly for any input).
- `expert-capture.cpp` + `build.sh` — **written**, compiles against our llama.cpp build
  (`b1-876a432`); runs on the VM next session to produce the real K2 numbers.
- Real K2 skew measurement + the "same model, less RAM at ≥X% speed" result — the headline
  experiment, next up. See [../PLAN.md](../PLAN.md) Phase 3.

> Honesty note: the skew number and RAM savings are **measured, not assumed**. If K2's routing
> turns out near-uniform on dev workloads, that is itself a publishable finding — and the
> chart title reports whatever the data shows.
