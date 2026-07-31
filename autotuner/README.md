# ExpertAtlas

MoE expert-activation profiler and placement autotuner.

Kimi K2 routes each token through 8 of 384 experts — and on real workloads the activation
distribution is heavily skewed. ExpertAtlas measures that skew on real developer traces,
then computes an optimal placement (hot experts in RAM, cold on NVMe) for a given memory
budget, emitted as llama.cpp tensor-offload configuration.

Goal: the same trillion-parameter model on a smaller, cheaper VM at near-full speed.

🚧 Lands in Phase 3 — see [PLAN.md](../PLAN.md).
