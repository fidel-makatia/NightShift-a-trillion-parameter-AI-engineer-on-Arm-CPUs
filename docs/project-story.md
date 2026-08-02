# NightShift — Project Story

## Inspiration

Everyone "knows" a trillion-parameter model needs a rack of GPUs. I didn't quite believe it,
because of one detail about how modern frontier models are built: they're **mixture-of-experts
(MoE)**. Kimi K2 has $1.04\text{T}$ total parameters but activates only about $32\text{B}$ per
token; Kimi K3 has $2.8\text{T}$ total and activates $\approx104\text{B}$. The full parameter set
never has to be *computed* at once — it only has to *fit in memory*.

That reframes the whole problem. Compute per token scales with the **active** parameters, not the
total:

$$\text{FLOPs}_{\text{token}} \approx 2 \cdot N_{\text{active}}$$

and memory *capacity* — not compute — becomes the gate. And memory capacity is exactly what
Arm-based cloud CPUs have, cheaply. Azure's **Cobalt 100** VMs give you $96$ Neoverse-N2 cores
and $660\text{ GiB}$ of RAM for roughly \$1/hour on spot — with **zero GPUs**. Microsoft is even
marketing Cobalt for "agentic AI workloads." So I asked: *can I run a trillion-parameter engineer
on one Arm CPU VM, and put it to work on the developer jobs where cost matters and latency
doesn't* — overnight PR review, issue triage, code generation? That became **NightShift**.

## What I built

NightShift is a one-command deployment (`terraform apply`) that stands up an OpenAI-compatible,
trillion-parameter endpoint on an Azure Cobalt 100 VM, plus the tooling to make it useful and the
optimization work to make it fast:

- **K2-in-a-box** — Terraform + cloud-init that provision the VM and a model disk, build
  llama.cpp with Arm's **KleidiAI** kernels, and serve Kimi K2/K3 (Unsloth dynamic GGUF quants).
- **The NightShift Action** — a GitHub Action that sends each pull request to the endpoint for an
  overnight review. It found real bugs in this repo's own code.
- **ExpertAtlas** — a C++ tool that hooks llama.cpp's graph eval callback, records which of the
  model's experts actually fire on real developer workloads, and emits a memory-placement policy.
- **A hand-written Arm kernel** — the compute core of MoE inference, optimized with Arm's `i8mm`
  **SMMLA** instruction, benchmarked head-to-head against llama.cpp's production kernels.
- **A reproducible playbook** — every benchmark, chart, and finding, regenerable from source.

## How I built it — and what the numbers said

I started by proving the thesis end to end: Kimi K2 (1.04T, 2-bit) loaded and generated on a
single `E96ps_v6` at **~9–11 tok/s**, with only ~15 GiB of *hot* RAM — the rest of the 360 GB of
weights sitting memory-mapped, paged as the router selected experts. That single measurement is
the whole idea in one number: **a trillion-parameter model is a memory-capacity problem, and Arm
cloud CPUs solve it.**

Then I pushed to the edge. Five days after Moonshot released **Kimi K3 (2.8T)**, I built llama.cpp
from its unmerged support PR and ran K3 on the same Arm VM — to my knowledge, the **first CPU
inference of K3 on Arm silicon**, at ~2.3 tok/s.

To make it genuinely useful, I had Kimi *build and run real software on the Arm CPU*: it wrote a
complete FastAPI microservice, which I then installed and served live (a working `307` redirect),
and — using continuous batching — I ran **four concurrent developer sessions on one VM** (an LRU
cache, a login UI, bare-metal Arm Cortex-M4 firmware, and a pytest suite), each with its own
prompt and measured tokens/sec.

## The math that shaped everything

Token generation is **memory-bandwidth-bound**: each token must read every active weight from RAM.
For K3 at $1.56$ bits/weight,

$$B_{\text{token}} = N_{\text{active}} \cdot \frac{\text{bits}}{8} \approx 104\text{B} \cdot \frac{1.56}{8} \approx 20\text{ GB/token},$$

so with a socket bandwidth of $\text{BW}\approx 200\text{ GB/s}$ the single-stream ceiling is

$$\text{tok/s}_{\max} \approx \frac{\text{BW}}{B_{\text{token}}} \approx \frac{200}{20} = 10\ \text{tok/s}.$$

I measured ~2.5, which told me the bottleneck at first was *not* the wall but **NUMA/threading
inefficiency** — the $E96$ spans two NUMA nodes of 48 cores, and running 88 threads across both
thrashed the interconnect. Fixing the thread count alone was a free $1.7\times$
($1.42 \to 2.45$ tok/s). But it also told me the honest truth: **20 tok/s single-stream on a 2.8T
model at this quant is above this machine's physics.** The real levers are batching, speculative
decoding, and more bandwidth (Cobalt 200) — not a magic flag.

## The optimization: a real Arm kernel

The point of an *optimization* challenge is to optimize something, so I wrote the operation at the
heart of every expert layer — the quantized GEMM — by hand for Neoverse-N2, using the **SMMLA**
(`i8mm`) instruction, a $2\times2$ int8 matrix-multiply:

$$C_{ij} \mathrel{+}= \sum_{k} A_{ik}\,B_{jk}, \qquad A \in \mathbb{Z}_8^{2\times8},\ B \in \mathbb{Z}_8^{2\times8}.$$

The first result was the most instructive: at batch $1$, my fancy kernel *tied* the compiler —
because a batch-1 GEMV is memory-bound, and no ALU trick helps. The win only appears when
**batching** raises arithmetic intensity and the op becomes compute-bound — which is *exactly* why
a shared, continuously-batched dev server is the right architecture. The kernel result and the
system design turned out to be the same story.

The prize was the K-quants. `Q2_K` is the format Kimi K2 actually runs, and **llama.cpp has no
`i8mm` GEMM for any K-quant** — it falls back to a per-row SDOT path. So I wrote the first one:
NEON-unpack the 2-bit weights to int8, run a $4\times4$-tiled SMMLA GEMM, and — the key insight —
keep the per-16-element scale correction *in vector registers* instead of extracting to scalars
(my first attempt did the scaling in scalars and lost all the gain at $0.97\times$). Linked
head-to-head against the real `libggml-cpu.so` on the VM, my kernel hit **41.9 vs 29.7 GFLOP/s —
$1.41\times$ over llama.cpp's production kernel, bit-exact.** That per-group scaling overhead is
also *why* ggml never added `i8mm` for K-quants — and now there's a path that does.

## What I learned

- **Memory bandwidth is the real currency of CPU LLM inference.** Almost every surprising result —
  the NUMA cliff, why fewer experts didn't help, why speculative decoding matters — falls out of
  the roofline once you accept that generation is bandwidth-bound.
- **MoE sparsity is what makes trillion-parameter CPU inference possible at all**, and the routing
  is measurably skewed on real workloads (ExpertAtlas: the hottest 50% of experts cover 76.5% of
  activations).
- **Arm's `i8mm`/SMMLA and SVE2 are genuinely powerful**, but a hand-written kernel only beats the
  compiler when you feed the instruction the right shape (batched) and keep the surrounding
  scale/dequant work vectorized.
- **Know the state of the art before you claim a win.** llama.cpp already SMMLA-optimizes Q8_0; the
  honest, novel contribution was the *un*optimized K-quant path.

## Why this matters

Frontier AI is effectively gatekept by GPUs — a single node capable of a trillion-parameter model
costs tens of thousands of dollars and is scarce. NightShift shows that the **same class of model
can run on one commodity Arm CPU VM for about \$1/hour**, entirely inside your own tenant, with
your code never leaving it. That is a real shift in access, cost, and privacy for the largest open
models — and it lands squarely on the Arm platform:

- **Access & cost.** A 1–2.8-trillion-parameter engineer becomes reachable to any developer with a
  cloud account, not just those with GPU budgets. A trillion-parameter PR review costs ~8 cents.
- **Privacy.** Self-hosted on your Arm VM — no code or data leaves your tenant, which GPU-API
  services can't offer.
- **A concrete contribution to the Arm ecosystem.** The **first `i8mm`/SMMLA GEMM for a K-quant**
  beats llama.cpp's production kernel by 1.41×, bit-exact — an upstreamable optimization that
  speeds up *every* Arm developer running the sub-2-bit quants that make these models fit in RAM.
- **A validated blueprint.** One `terraform apply` reproduces the whole system, and the kernel
  benchmark validates on any Arm machine in seconds — this isn't a one-off demo, it's a repeatable
  path for running frontier MoE models on Arm CPUs.

## Challenges I faced

- **Azure capacity.** Mid-project, the region ran out of Arm VM capacity — `E96`, `E48`, and `E32`
  all failed to allocate in the zone I was in. I made the model disk snapshot-restorable and
  zone-portable in Terraform and rebuilt in another zone (re-downloading the 360 GB model).
- **A VPN that kept breaking SSH.** For a long stretch every command timed out. The cause turned
  out to be my own VPN changing my egress IP — and the VM's firewall only allowed one IP. Now I
  know: if SSH hangs, check the NSG rule first.
- **Bleeding-edge K3.** The K3 support PR is unmerged and the server was unstable under load —
  crashes, "router mode," empty responses. Moving it under `systemd` (which is why the K2 server
  had been rock-solid) finally stabilized it.
- **The bandwidth wall itself.** I wanted 20 tok/s and had to accept that single-stream on a 2.8T
  model isn't reachable on Cobalt 100 — and to say so honestly rather than inflate a number.
- **The kernel.** Beating a shipping, well-tuned inference engine is hard; the first two attempts
  matched or lost. The breakthrough was diagnosing *where* the SMMLA advantage was being spent
  (scalar scale extraction) and moving that work into vector registers.

## What's next

Wire ExpertAtlas's placement policy in to fit K3 on a smaller VM; upstream the K-quant SMMLA path
to llama.cpp; add speculative decoding for Kimi; and benchmark Cobalt 200 (+50% bandwidth) the
moment it's available. NightShift is open source (Apache-2.0) — one `terraform apply`, and your
code never leaves your tenant.
