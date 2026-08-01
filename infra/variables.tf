variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "location" {
  description = "Azure region with Epsv6 (Cobalt 100) capacity"
  type        = string
  default     = "eastus2"
}

variable "zone" {
  description = "Availability zone (PremiumV2 disks require zonal deployment)"
  type        = string
  default     = "2"
}

variable "vm_size" {
  description = "Arm VM size. E96ps_v6 = 96 vCPU / 672 GiB (full K2). E32ps_v6 = 32 vCPU / 256 GiB (ExpertAtlas target)."
  type        = string
  default     = "Standard_E96ps_v6"
}

variable "use_spot" {
  description = "Use Spot pricing (requires lowPriorityCores quota)"
  type        = bool
  default     = false
}

variable "model_disk_gb" {
  description = "Model disk size. K2 UD-Q2_K_XL ~360GB + headroom for a second quant."
  type        = number
  default     = 1024
}

variable "model_snapshot_id" {
  description = "If set, restore the model disk from this snapshot (enables zone moves without re-downloading)."
  type        = string
  default     = ""
}

variable "admin_username" {
  type    = string
  default = "nightshift"
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "allowed_ip" {
  description = "CIDR allowed to reach SSH and the llama-server port"
  type        = string
}
