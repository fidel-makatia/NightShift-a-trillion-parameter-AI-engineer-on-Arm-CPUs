# The NightShift Action

A GitHub Action that turns your [K2-in-a-box](../infra/) endpoint into an **overnight
engineer**: every pull request gets reviewed by a trillion-parameter model running on your
own Arm CPUs, posted as a PR comment — and your code never leaves your tenant.

## Why async CPU inference is the right tool here

CPU inference is slow per token (~10 tok/s for K2) but costs cents per task. PR review is
**asynchronous** — nobody watches tokens stream while a bot reviews a PR at 2am — so the only
weakness that matters (latency) doesn't, and the strengths (cost, privacy) do. A full deep
review costs about **8 cents** on spot pricing; an overnight batch of 100 is about **$8**.

## Usage

1. Deploy the endpoint: `cd ../infra && terraform apply`.
2. Add `NIGHTSHIFT_ENDPOINT` as a repo secret (keep the server private — Tailscale, a
   self-hosted runner on the same VNet, or a tunnel).
3. Copy [`example-workflow.yml`](example-workflow.yml) to `.github/workflows/nightshift.yml`.

That's it — open a PR and NightShift comments on it.

## It found real bugs in this repo

Pointed at our own PR #7 (the benchmark launcher scripts), NightShift flagged genuine
defects we'd shipped: a missing restart-on-failure path for the inference server, and a
launcher that never verified the remote job started. See
[`../playbook/artifacts/shot_devjob_qwen.png`](../playbook/artifacts/shot_devjob_qwen.png).

## Files

- [`action.yml`](action.yml) — composite action definition
- [`review.sh`](review.sh) — fetch diff → call model → post comment (safe jq request building)
- [`example-workflow.yml`](example-workflow.yml) — drop-in workflow
