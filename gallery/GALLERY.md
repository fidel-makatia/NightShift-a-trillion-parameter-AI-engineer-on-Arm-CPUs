# NightShift — image gallery (Devpost-ready)

16 curated images in submission order. Captions are written to paste straight into the
Devpost gallery. All are real captures/renders from the Azure Cobalt 100 (Neoverse-N2) VM.

| # | File | Devpost caption |
|---|------|-----------------|
| 01 | `01_kimi-k3-2.8T-on-arm.png` | Kimi K3 — **2.8 trillion parameters** — generating on Arm CPUs (Azure Cobalt 100), no GPU. First known CPU inference of K3 on Arm silicon. |
| 02 | `02_the-hardware-no-gpu.png` | The machine: 96 Arm Neoverse-N2 cores, 660 GiB RAM, and `nvidia-smi: command not found` — there is no GPU. |
| 03 | `03_first-words-trillion-params.png` | A trillion-parameter model's first words on Arm — a haiku, generated live at ~7.5 tok/s, zero GPUs. |
| 04 | `04_live-microservice-on-arm.png` | A real dev job: Kimi builds a FastAPI microservice, and it runs and serves live traffic (307 redirect) on the Arm CPU. |
| 05 | `05_kimi-generated-code.png` | The 73 lines of production FastAPI that Kimi K2 wrote on the Arm CPU at 8.6 tok/s. |
| 06 | `06_four-concurrent-dev-sessions.png` | One Arm VM, four concurrent dev sessions on Kimi K3 — each with its prompt and its tokens/sec: LRU cache, login UI, Cortex-M4 firmware, pytest suite. |
| 07 | `07_webapp-ui-desktop.png` | The login page Kimi K3 wrote, actually rendered (desktop) — dark theme, gradient sign-in button. |
| 08 | `08_webapp-ui-mobile.png` | The same K3-generated web app, responsive on mobile. |
| 09 | `09_pr-review-deep-tier.png` | NightShift reviewing a real pull request on the deep tier (Kimi K2, 1.04T) — it caught genuine bugs, at ~$0.08 per review. |
| 10 | `10_pr-review-interactive-tier.png` | The same review on the interactive tier (Qwen3-30B) — 7 issues found at 42 tok/s. |
| 11 | `11_arm-kernel-q2k-beats-llamacpp.png` | **The headline optimization:** the first SMMLA (i8mm) GEMM for a K-quant. On Q2_K — the format Kimi K2 runs — it beats llama.cpp's production kernel **1.41×**, bit-exact. |
| 12 | `12_arm-kernel-q8-head-to-head.png` | My hand-written Arm kernel vs llama.cpp's production Q8_0 kernel, linked head-to-head on Neoverse-N2: 94.9 vs 37.9 GFLOP/s, bit-exact. |
| 13 | `13_two-tiers-one-vm.png` | Two model tiers on one Arm VM: K2/K3 for deep async work, Qwen3-30B for interactive coding at ~48 tok/s. |
| 14 | `14_cost-per-review.png` | The economics: a trillion-parameter code review costs ~8 cents on spot pricing; 100 overnight ≈ $8. |
| 15 | `15_bandwidth-wall.png` | The core finding: generation is memory-bandwidth-bound — throughput peaks at 64 threads and drops at 96 (the NUMA wall). |
| 16 | `16_expertatlas-routing-skew.png` | ExpertAtlas: measured expert-routing skew on real dev workloads — the hottest 50% of experts cover 76.5% of activations. |

**Cover image suggestion:** `01_kimi-k3-2.8T-on-arm.png` (the wow) or `11_arm-kernel-q2k-beats-llamacpp.png` (the engineering).
