## Inspiration

Everyone "knows" a trillion-parameter model needs a rack of GPUs. I didn't quite believe it —
because modern frontier models are **mixture-of-experts (MoE)**. Kimi K2 has $1.04\text{T}$
parameters but activates only $\sim32\text{B}$ per token; Kimi K3 has $2.8\text{T}$ and activates
$\sim104\text{B}$. The full model never has to be *computed* at once — it only has to *fit in
memory*. Compute per token scales with the **active** params, not the total:

$$\text{FLOPs}_{\text{token}} \approx 2 \cdot N_{\text{active}}.$$

So memory *capacity* becomes the gate — and that's exactly what Arm cloud CPUs have, cheaply. An
Azure **Cobalt 100** VM gives you $96$ Neoverse-N2 cores and $660\text{ GiB}$ of RAM for ~\$1/hour
on spot, with **zero GPUs**. I asked: can I run a trillion-parameter AI engineer on one Arm CPU
VM and put it to work on the developer jobs where cost matters and latency doesn't — overnight PR
review, issue triage, code generation? That became **NightShift**.

## What it does

NightShift turns one Arm CPU VM into a private, trillion-parameter dev engineer:

- **One-command deploy** — `terraform apply` stands up an OpenAI-compatible endpoint serving
  Kimi K2 (1.04T) or K3 (2.8T) via llama.cpp + Arm KleidiAI, no GPU.
- **Real dev work** — it writes and *runs* software on the Arm CPU (a live FastAPI microservice),
  reviews pull requests overnight (a GitHub Action that found real bugs in this repo), and serves
  **four concurrent dev sessions at once** via continuous batching.
- **Two tiers, one VM** — K2/K3 for deep async work; Qwen3-30B for interactive coding at ~48 tok/s.
- **A hand-written Arm kernel** — the first `i8mm`/SMMLA GEMM for a K-quant, beating llama.cpp's
  production kernel **1.41×**, bit-exact.
- **A reproducible playbook** — every benchmark and chart, regenerable from source. Your code
  never leaves your tenant; a trillion-parameter PR review costs about **8 cents**.

## How we built it

I began by proving the thesis: Kimi K2 (2-bit) generating at **~9–11 tok/s** on a single
`E96ps_v6`, with only ~15 GiB *hot* RAM while its 360 GB of weights stayed memory-mapped. Then I
pushed to the frontier — five days after Moonshot released **Kimi K3 (2.8T)**, I built llama.cpp
from its unmerged support PR and ran it on the same Arm box: to my knowledge the **first CPU
inference of K3 on Arm silicon** (~2.3 tok/s).

The stack: Terraform + cloud-init provision the VM, a PremiumV2 model disk, and a llama.cpp build
with `-DGGML_CPU_KLEIDIAI=ON`; Unsloth dynamic GGUF quants; a benchmark harness; **ExpertAtlas**, a
C++ tool that hooks llama.cpp's graph eval callback to record which experts fire on real dev
traffic; a GitHub Action for PR review; and a hand-written Arm kernel. That kernel targets the
quantized GEMM at the heart of every expert layer using **SMMLA** (a $2\times2$ int8
matrix-multiply):

$$C_{ij} \mathrel{+}= \sum_{k} A_{ik}\,B_{jk},\qquad A,B \in \mathbb{Z}_8^{2\times8}.$$

## Challenges we ran into

- **The bandwidth wall.** Generation is memory-bandwidth-bound: $B_{\text{token}} \approx 20$ GB
  for K3, so at $\sim200$ GB/s the ceiling is $\text{tok/s}_{\max}\approx\text{BW}/B_{\text{token}}
  \approx 10$. I *wanted* 20 tok/s single-stream and had to accept — honestly — that it's above
  Cobalt 100's physics, and report the real levers (batching, speculative decoding, Cobalt 200).
- **Azure capacity.** Mid-project the region ran dry of Arm VMs — `E96`, `E48`, and `E32` all
  failed to allocate in my zone. I made the model disk snapshot-restorable and zone-portable in
  Terraform and rebuilt elsewhere.
- **A VPN that kept breaking SSH.** Every command timed out for a stretch; the cause was my VPN
  changing my egress IP while the VM firewall allowed only one.
- **Bleeding-edge K3.** The unmerged support PR was unstable under load until I moved the server
  under `systemd`.
- **Beating a production kernel.** My first SMMLA kernel only *matched* llama.cpp ($0.97\times$) —
  the per-16-element scale correction was being done in scalars. Moving it into vector registers
  unlocked the win.

## Accomplishments that we're proud of

- **First known CPU inference of Kimi K3 (2.8T) on Arm silicon** — five days after release.
- **A hand-written Arm kernel that beats llama.cpp's production code.** On `Q2_K` (the quant K2
  runs), which had *no* `i8mm` path in ggml, my SMMLA GEMM hits **41.9 vs 29.7 GFLOP/s (1.41×),
  bit-exact** against the linked `libggml-cpu.so`.
- **A trillion-parameter model doing real, runnable work** — building and serving a live
  microservice, and four concurrent dev sessions, on one CPU VM.
- **Honest, reproducible economics:** ~8¢ per deep PR review; one `terraform apply` to reproduce
  the whole thing.

## What we learned

- **Memory bandwidth is the real currency of CPU LLM inference** — the NUMA cliff, the "fewer
  experts didn't help" result, and why speculative decoding matters all fall out of the roofline.
- **MoE sparsity is what makes trillion-parameter CPU inference possible**, and routing is
  measurably skewed (ExpertAtlas: the hottest 50% of experts cover 76.5% of activations).
- **Arm's `i8mm`/SMMLA is powerful, but only with the right shape** — batched, with the
  dequant/scale work kept vectorized — and llama.cpp already optimizes Q8_0, so the honest, novel
  contribution was the un-optimized K-quant path.
- I also found [Colibri](https://github.com/JustVugg/colibri), an x86 engine proving this thesis —
  and the gap it leaves open is Arm, which is exactly where NightShift lives.

## What's next for NightShift — a trillion-parameter AI engineer on Arm CPUs

Wire ExpertAtlas's placement policy in to fit K3 on a smaller, cheaper VM; upstream the K-quant
SMMLA kernel to llama.cpp; add speculative decoding for Kimi (the one lever that beats the
per-token bandwidth wall); and benchmark **Azure Cobalt 200** (+50% memory bandwidth) the moment
it's available. NightShift is open source (Apache-2.0) — one `terraform apply`, and your code
never leaves your tenant.
