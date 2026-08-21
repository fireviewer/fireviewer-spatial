variable "aws_region" {
  description = "AWS region. The first deployment is locked to Paris."
  type        = string
  default     = "eu-west-3"

  validation {
    condition     = var.aws_region == "eu-west-3"
    error_message = "The reference deployment is locked to eu-west-3 (Paris)."
  }
}

variable "aws_profile" {
  description = "Local AWS CLI profile. CI leaves this null and uses OIDC."
  type        = string
  default     = null
  nullable    = true
}

variable "name_prefix" {
  description = "Prefix applied to Map Builder resources."
  type        = string
  default     = "fireviewer-map-builder"
}

variable "instance_type" {
  description = "Fixed G2/G3 compute profile. Capacity failures must stop instead of scaling up."
  type        = string
  default     = "m7i-flex.large"

  validation {
    condition     = var.instance_type == "m7i-flex.large"
    error_message = "Only m7i-flex.large is authorized for the reference deployment."
  }
}

variable "root_volume_size_gib" {
  description = "Small operating-system volume; builder work belongs on /scratch."
  type        = number
  default     = 20

  validation {
    condition     = var.root_volume_size_gib >= 16 && var.root_volume_size_gib <= 30
    error_message = "The root volume must remain between 16 and 30 GiB."
  }
}

variable "scratch_volume_size_gib" {
  description = "Ephemeral gp3 volume mounted at /scratch."
  type        = number
  default     = 100

  validation {
    condition     = var.scratch_volume_size_gib >= 80 && var.scratch_volume_size_gib <= 100
    error_message = "The scratch volume must remain between 80 and 100 GiB."
  }
}

variable "scratch_iops" {
  description = "Baseline gp3 IOPS."
  type        = number
  default     = 3000

  validation {
    condition     = var.scratch_iops == 3000
    error_message = "The initial gp3 profile is fixed at 3000 IOPS."
  }
}

variable "scratch_throughput_mibps" {
  description = "Baseline gp3 throughput in MiB/s."
  type        = number
  default     = 125

  validation {
    condition     = var.scratch_throughput_mibps == 125
    error_message = "The initial gp3 profile is fixed at 125 MiB/s."
  }
}

variable "cache_retention_days" {
  description = "Retention for transient S3 cache objects."
  type        = number
  default     = 14

  validation {
    condition     = var.cache_retention_days >= 7 && var.cache_retention_days <= 30
    error_message = "Cache retention must remain between 7 and 30 days."
  }
}

variable "monthly_budget_usd" {
  description = "Gross monthly AWS cost ceiling used for notifications."
  type        = number
  default     = 10

  validation {
    condition     = var.monthly_budget_usd > 0 && var.monthly_budget_usd <= 100
    error_message = "The budget must be greater than zero and no more than the available initial credits."
  }
}

variable "budget_alert_email" {
  description = "Optional email subscribed to the 25/50/75/85/90 percent budget alerts."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.budget_alert_email == null || can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.budget_alert_email))
    error_message = "budget_alert_email must be null or a valid email address."
  }
}

variable "enable_batch" {
  description = "Creates Batch resources only after G2/G3 approval. Must remain false now."
  type        = bool
  default     = false
}

variable "g2_validated" {
  description = "Explicit gate proving the direct EC2 build passed semantic and resource validation."
  type        = bool
  default     = false
}

check "batch_requires_g2" {
  assert {
    condition     = !var.enable_batch || var.g2_validated
    error_message = "AWS Batch cannot be enabled before direct EC2 gates G2 and G3 pass."
  }
}
