# The NightShift Action

A GitHub Action that turns your K2-in-a-box endpoint into an overnight engineer:

- **PR review** — every opened PR gets a trillion-parameter review by morning
- **Issue triage** — labels, duplicates, severity
- **Changelog** — nightly draft from merged PRs

Async by design: CPU inference is slow per token but costs cents per task, and nobody
watches tokens stream at 2am. Code never leaves your Azure tenant.

🚧 Lands in Phase 4 — see [PLAN.md](../PLAN.md).
