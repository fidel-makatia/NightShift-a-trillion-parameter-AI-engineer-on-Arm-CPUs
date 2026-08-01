# NightShift 🌙

**A 1-trillion-parameter AI engineer that runs entirely on Arm CPUs — no GPU on Earth involved.**

NightShift runs [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) — a 1.04-trillion-parameter
mixture-of-experts model — on spot-priced, Arm-based **Azure Cobalt 100** VMs, and puts it to
work on the developer tasks where cost matters and latency doesn't: reviewing your pull
requests, triaging issues, and fixing bugs overnight. For about **$1/hour**, on hardware with
zero GPUs, with your code never leaving your own Azure tenant.

> Built for the [Arm Create: AI Optimization Challenge 2026](https://arm-ai-optimization-challenge.devpost.com/) — Cloud AI track.

## Why this works (the one-paragraph version)

Trillion-parameter models are assumed to need a GPU cluster. But K2 is a mixture-of-experts
model: 1T total parameters, only **32B active per token** — so per-token compute is that of a
32B model, while the 1T just has to *fit in memory*. Azure's Arm-based Cobalt 100 VMs
(E96ps_v6: 96 cores, 672 GiB RAM) have that memory, and Arm's KleidiAI kernels (i8mm, SVE)
make quantized CPU inference fast enough for asynchronous work. NightShift measures exactly
where the limits are, then engineers around them with **expert-placement autotuning** — profiling
which of K2's 384 experts actually fire on real workloads and placing hot experts in RAM,
cold ones on NVMe, so the model fits on smaller, cheaper VMs.

## What's in this repo

| Component | What it is |
|---|---|
| [`infra/`](infra/) | **K2-in-a-box** — one `terraform apply` gives you an OpenAI-compatible, trillion-parameter endpoint on an Azure Cobalt VM |
| [`autotuner/`](autotuner/) | **ExpertAtlas** — MoE expert-activation profiler + placement autotuner (the novel engineering) |
| [`bench/`](bench/) | Reproducible benchmark harness: tokens/sec, TTFT, throughput, quality evals across quants, KleidiAI on/off, scale-up vs scale-out |
| [`action/`](action/) | **The NightShift Action** — GitHub Action that sends PRs to your endpoint for overnight review, issue triage, changelog generation |
| [`playbook/`](playbook/) | **The Trillion-Parameter CPU Playbook** — the full study, charts, and how-to |

## Quickstart

> 🚧 Under construction — challenge submission deadline is Aug 14, 2026. See [PLAN.md](PLAN.md).

```bash
# The goal:
cd infra && terraform apply        # ~15 min: Cobalt VM + model disk + llama.cpp server
export OPENAI_BASE_URL=http://<vm-ip>:8080/v1
aider --model openai/kimi-k2      # a trillion-parameter coding agent, on your own tenant
```

## The numbers (first measurements — 2026-07-31)

**It works.** Kimi K2 Thinking — 1.04 trillion parameters — running on a single Azure
E96ps_v6 (96× Arm Neoverse-N2 cores, Cobalt 100, 672 GiB RAM, **zero GPUs**):

| Metric | Measured |
|---|---|
| Token generation | **8.98 tok/s** (warm), 6.09 tok/s first run |
| Prompt processing | 29.3 tok/s |
| Time to first token | 479 ms (14-token prompt) |
| Model footprint | 360 GB on disk (Unsloth UD-Q2_K_XL dynamic 2-bit) |
| Resident memory under load | ~15 GiB hot + page-cached weights — the MoE sparsity story in one number |
| GPUs involved | 0 |

![Thread scaling on Cobalt 100](playbook/artifacts/threads_sweep.png)

**First playbook finding:** prompt processing (compute-bound) scales with cores —
6.6 → 12.7 → 20.6 tok/s at 24/48/96 threads. Token generation (bandwidth-bound) **peaks at
48 threads (10.4 tok/s) and *drops* at 96**: the E96ps_v6 spans two NUMA nodes of 48 cores,
and generation saturates one node's memory bandwidth. NUMA-aware placement, not more cores,
is the lever — exactly the kind of Arm-specific tuning the playbook exists to document.

Raw artifacts: [`playbook/artifacts/`](playbook/artifacts/) — quant ladder, KleidiAI on/off,
and NUMA-pinned runs land next; see [PLAN.md](PLAN.md) Phase 2.

## License

[Apache 2.0](LICENSE)
