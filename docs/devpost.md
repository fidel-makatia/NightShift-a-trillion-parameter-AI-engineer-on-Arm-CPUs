# Devpost Submission Draft — NightShift

> Working draft of every text field the Devpost form needs. Update numbers as benchmarks land.

## Project name

**NightShift — a trillion-parameter AI engineer on Arm CPUs**

## Elevator pitch (tagline — 189 chars, fits Devpost's 200-char limit)

> A 1-trillion-parameter AI engineer that reviews your PRs and fixes bugs overnight — on Arm
> CPUs, no GPU on Earth involved. ~$1/hour on Azure Cobalt, and your code never leaves your tenant.

**Even shorter fallback (120 chars):**

> A trillion-parameter AI engineer that reviews your PRs overnight — on Arm CPUs, no GPU,
> ~$1/hour, on your own tenant.

## Inspiration

Everyone "knows" trillion-parameter models need GPU clusters. But mixture-of-experts models
broke the assumption behind that: Kimi K2 has 1.04T parameters and activates only 32B per
token. The 1T just has to *fit in memory* — and memory is exactly what Arm-based cloud CPUs
have in abundance, at spot prices GPUs will never touch. Meanwhile, half the AI work a dev
team wants is *asynchronous* — nobody watches tokens stream at 2am when a bot reviews a PR.
We put those two facts together.

## What it does

NightShift is four things stacked into one story:

1. **K2-in-a-box** (`infra/`): one `terraform apply` provisions an Azure Cobalt 100 VM
   (Arm Neoverse), attaches a pre-built model disk, and serves Kimi K2 through an
   OpenAI-compatible endpoint via llama.cpp built with Arm KleidiAI kernels.
2. **ExpertAtlas** (`autotuner/`): our novel engineering. We profile which of K2's 384
   experts actually activate on real developer workloads (code review, bug fixing, chat),
   show the distribution is heavily skewed, and exploit it — hot experts pinned in RAM,
   cold experts on NVMe — so the trillion-parameter model runs on smaller, cheaper VMs
   at near-full speed.
3. **The NightShift Action** (`action/`): a GitHub Action that turns the endpoint into an
   overnight engineer — PR reviews, issue triage, changelog drafts — plus the same endpoint
   wired into Aider/Cline for interactive use. Private by construction: code never leaves
   your tenant.
4. **The Trillion-Parameter CPU Playbook** (`playbook/`): the reproducible study — quant
   ladder (1-bit → 4-bit) measured for speed *and* quality, KleidiAI/i8mm on vs off,
   scale-up vs scale-out, MoE sparsity comparison (K2 32B-active vs Llama 4 Maverick
   17B-active), thread/NUMA sweeps. Every chart regenerable from `bench/`.

## How we built it

<!-- fill in as built: llama.cpp + KleidiAI build flags, Terraform, GGUF quants (Unsloth
dynamic), expert-activation instrumentation approach, GitHub Action architecture -->

## Challenges we ran into

<!-- fill in honestly: quota, 600GB downloads, RPC maturity, prompt-processing wall,
what the expert-activation data actually showed -->

## Accomplishments we're proud of

- First public CPU-only deployment of a 1T-parameter model on Arm server silicon
- Expert-placement autotuning: <!-- X% --> smaller memory footprint at <!-- Y% --> of full speed
- $<!-- Z --> per million tokens / $<!-- W --> per PR review, measured
- Upstream contribution: <!-- PR link -->

## What we learned

<!-- the playbook's headline findings, condensed -->

## What's next

Cobalt 200 (50% faster, in preview) numbers; expert-parallel sharding across a spot-VM
swarm; packaging ExpertAtlas for any MoE model, not just K2.

## Built with

`arm` `azure-cobalt` `llama.cpp` `kleidiai` `terraform` `kimi-k2` `python` `github-actions`

## Judging-criteria checklist (internal — delete before submitting)

- [ ] Tech (40): KleidiAI/i8mm leverage measured; ExpertAtlas is original engineering; clean repo
- [ ] Wow (25): 1T-no-GPU headline; ≤3-min video: agent fixes a bug + reviews this repo's own PR
- [ ] Impact (20): playbook is reusable research; Action + Terraform reusable by any team; upstream PR
- [ ] DX (15): one-command deploy; step-by-step Arm64 setup docs; README quickstart tested from scratch
- [ ] Compliance: public repo, Apache-2.0 LICENSE visible at repo top, setup instructions for Arm64, video hosted on YouTube
