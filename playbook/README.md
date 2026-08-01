# The Trillion-Parameter CPU Playbook

**How to serve a 1-trillion-parameter model on Arm CPUs — what works, what doesn't,
and what it costs.** Every number here is measured on Azure Cobalt 100 and reproducible
from [`../bench/`](../bench/). No GPUs were used anywhere in this study.

- **Hardware:** Azure `Standard_E96ps_v6` — 96× Arm Neoverse-N2 (Cobalt 100), 672 GiB RAM,
  two NUMA nodes of 48 cores, eastus2
- **Runtime:** llama.cpp (`b1-876a432`) built with `-DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=ON`
- **Models:** Kimi K2 Thinking (1.04T params, Unsloth UD-Q2_K_XL 2-bit dynamic, 360 GB);
  Qwen3-30B-A3B (Q4_K_M, 18.5 GB) as the interactive tier

---

## TL;DR for the impatient

| | Kimi K2 (1.04T) | Qwen3-30B-A3B |
|---|---|---|
| Active params / token | 32B | 3B |
| Token generation (best config) | **11.0 tok/s** @ 64 threads | **48.0 tok/s** @ 48 threads |
| Prompt processing | 20.6 tok/s @ 96 threads | 232 tok/s @ 96 threads |
| Cost per PR review (spot) | **$0.081** | $0.012 |
| Role | deep async work (overnight review) | interactive coding |

The trillion-parameter model runs on one CPU VM because it's mixture-of-experts:
only 32B of the 1.04T parameters activate per token, so per-token *compute* is that of a
32B model — the trillion just has to fit in RAM, which Cobalt VMs have.

---

## Finding 1 — It works, and memory is the enabling resource, not compute

Kimi K2 Thinking loads and serves from a single E96ps_v6. Resident working set under load
is ~15 GiB; the 360 GB of weights sit in the page cache, memory-mapped. This is the whole
thesis in one measurement: **a trillion-parameter model is a memory-capacity problem, and
Arm cloud CPUs solve it cheaply.**

## Finding 2 — More cores ≠ more tokens (the bandwidth wall)

Token generation is memory-bandwidth-bound. It does **not** scale with core count:

| Threads | K2 generation | Qwen3 generation |
|---|---|---|
| 24 | 7.4 tok/s | 47.4 tok/s |
| 48 | 10.4 tok/s | **48.0 tok/s** |
| 64 | **11.0 tok/s** | — |
| 96 | 9.6 tok/s | 3.2 tok/s |

Prompt processing (compute-bound) *does* scale — K2 goes 6.6 → 12.7 → 20.6 tok/s at
24/48/96 threads. The lesson: **tune threads per phase**, and never assume all 96 cores help.

![NUMA cliff](artifacts/numa_cliff.png)

## Finding 3 — NUMA topology dominates everything

The E96ps_v6 is two 48-core NUMA nodes. Cross-node memory traffic is the single biggest
performance factor we found:

- `--numa distribute` across both nodes: **2.3 tok/s** — 4.8× *slower* than leaving it alone.
- Qwen3 at 96 threads (spanning both nodes) collapses to 3.2 tok/s from 48 tok/s at one node.
- The 2-bit K2 (360 GB) is too large to strict-bind to a single 336 GB node; a 1-bit quant
  (245 GB) fits and is the path to single-node generation (next experiment).

**Deployment rule:** keep a model's generation threads inside one NUMA node.

## Finding 4 — Don't co-locate two polling inference servers

Running the K2 and Qwen3 servers simultaneously on the same unpinned core pool dropped K2
from ~10 tok/s to **1.5 tok/s** — both servers busy-poll and fight for cores and cache.
The fix is the NightShift two-tier deployment: **pin the deep-tier server to NUMA node 0 and
the interactive-tier server to node 1** (`numactl --cpunodebind`), so they never contend.
(We hit this the honest way — debugging it live is documented in the commit history.)

## Finding 5 — The economics are the real story

At measured throughput and Azure retail spot pricing ($1.15/hr for the whole 96-core VM):

- A trillion-parameter deep PR review (3K-token diff → 1.2K-token review): **~8 cents**.
- An overnight batch of 100 such reviews: **~$8**.
- The same review on the interactive tier: **~1 cent**.

![Cost per review](artifacts/cost_per_review.png)

---

## Reproduce it

```bash
cd infra && terraform apply          # provisions the VM + model disk, builds llama.cpp+KleidiAI
./infra/vm.sh ssh                    # onto the box
bash bench/bench.sh <model.gguf>     # thread sweep → CSV
python3 bench/plot_threads.py        # CSV → chart
```

## Open threads (tracked in [PLAN.md](../PLAN.md))

- 1-bit K2 pinned to a single NUMA node — can we beat 11 tok/s?
- KleidiAI on vs off, isolated — quantifying the Arm-kernel contribution
- **ExpertAtlas**: profiling which of K2's 384 experts fire on real dev workloads, to fit
  the model in less RAM (the project's headline engineering contribution)
- Azure Cobalt 200 (50% more bandwidth, in preview) — the obvious next hardware step
