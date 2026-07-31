# The Trillion-Parameter CPU Playbook

The reproducible study behind NightShift. Every chart regenerates from [`bench/`](../bench/).

Questions it answers:

1. How fast is 1T-parameter (32B-active) inference on Azure Cobalt 100, really?
   (tokens/sec, TTFT, prompt-processing rate, $/Mtok)
2. What do Arm KleidiAI/i8mm kernels buy you? (on vs off, same workload)
3. Which quant is the sweet spot? (1→4-bit, speed **and** quality measured)
4. How much does MoE sparsity matter? (K2 32B-active vs Llama 4 Maverick 17B-active)
5. Scale-up vs scale-out: one big VM vs an RPC swarm of small ones
6. How skewed is expert activation on real dev workloads, and how much RAM can
   expert placement save? (→ [ExpertAtlas](../autotuner/))

🚧 Fills in through Phases 2–3 — see [PLAN.md](../PLAN.md).
