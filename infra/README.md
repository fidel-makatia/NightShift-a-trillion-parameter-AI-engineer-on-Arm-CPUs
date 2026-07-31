# K2-in-a-box

One `terraform apply` → an OpenAI-compatible, trillion-parameter endpoint on an Azure
Cobalt 100 (Arm) VM.

Provisions: E96ps_v6 spot VM · 1TB Premium SSD v2 model disk (or snapshot clone) · NSG ·
cloud-init that builds llama.cpp with KleidiAI/i8mm and starts `llama-server`.

🚧 Terraform lands in Phase 1 — see [PLAN.md](../PLAN.md).
