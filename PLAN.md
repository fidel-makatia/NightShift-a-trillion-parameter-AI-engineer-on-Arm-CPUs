# NightShift — 14-Day Build Plan

Deadline: **Aug 14, 2026, 4:00 PM PDT** (submit Aug 13; the last day is buffer).
Strategy: retire the boring risks first (quota, download), bank the guaranteed demo
(single-VM 1T inference) by day 4, then spend everything else on the differentiators
(ExpertAtlas, playbook, Action).

## Phase 0 — Today (Thu Jul 31): unblock everything

- [ ] **Azure quota request** for Epsv6 spot vCPUs (ask for 192; region with Epsv6 + good spot
      capacity, e.g. East US / West US 2). This is the long pole — file it first.
- [x] Repo scaffold, license, pitch, plan
- [ ] Register/submit draft on Devpost; join Arm Discord (office hours = free judge signal)
- [ ] Create GitHub repo (public), push scaffold

## Phase 1 — Fri Aug 1 → Sun Aug 3: infrastructure + model on disk

- [ ] Terraform: E96ps_v6 spot VM + 1TB Premium SSD v2 + NSG, cloud-init installs toolchain
- [ ] Build llama.cpp on the VM with KleidiAI/i8mm flags; verify with a small model first
- [ ] Download Unsloth K2 Thinking UD-Q2_K_XL (~360 GB) to the data disk; **snapshot the disk**
      so every future VM (respins, swarm, smaller sizes) attaches a copy instead of re-downloading
- [ ] Milestone: **first token from 1T params on Arm** — screenshot everything

## Phase 2 — Mon Aug 4 → Wed Aug 6: baseline benchmarks (the playbook's raw material)

- [ ] Bench harness (`bench/`): tokens/sec, TTFT, prompt-processing rate, RAM, $/Mtok
- [ ] Sweeps: quant ladder (1/2/3/4-bit) × KleidiAI on/off × thread counts
- [ ] Quality evals per quant (small fixed suite: coding + reasoning tasks) — speed without
      quality is a toy; this chart is what makes it research
- [ ] Comparison points: quant ladder; note Cobalt 200 preview if granted

## Phase 3 — Thu Aug 7 → Sun Aug 10: ExpertAtlas (the winning engineering)

- [ ] Instrument expert activation per layer on real traces (PR diffs, bug-fix sessions, chat)
- [ ] Analyze skew → placement policy (hot experts RAM, cold NVMe via llama.cpp MoE tensor offload)
- [ ] Autotuner: given a RAM budget, emit optimal placement; validate on a smaller/cheaper VM
      (target headline: "same model, half the RAM, ≥80% of the speed")
- [ ] Upstream PR opportunity: whatever we hit (llama.cpp Arm build, RPC fix, docs) — open it
- [ ] Stretch: 2-node RPC swarm for the Q4 quant (scale-out data point; cut without mercy if it fights)

## Phase 4 — Fri Aug 8 → Tue Aug 12 (overlaps P3): the developer product

- [ ] `action/`: GitHub Action — PR opened → K2 review posted (works against our own repo = demo)
- [ ] Issue triage + changelog modes
- [ ] Wire Aider/Cline to the endpoint; record a real bug-fix session
- [ ] Harden K2-in-a-box: `terraform apply` → endpoint, documented end-to-end from a clean machine

## Phase 5 — Tue Aug 12 → Wed Aug 13: ship

- [ ] Playbook writeup with final charts (regenerable from bench/)
- [ ] 3-min video: cold open on the terminal ("1T params, 0 GPUs"), agent fixes a real bug,
      NightShift reviews this repo's own PR, cost slide, done. Time-lapse anything slow.
- [ ] README polish; quickstart re-tested from scratch; LICENSE visible; **submit Aug 13**

## Risk ladder (pre-decided fallbacks)

1. Quota denied for 96 vCPU → 2× E48ps_v6 or E64ps_v6 + 1-bit quant (245 GB)
2. RPC swarm flaky → cut it; single-VM story is complete without it
3. ExpertAtlas placement gains disappoint → the *profiling study* alone is still novel content
4. K2 Thinking too slow for video → K2.7 Code for the live demo, K2 for async
