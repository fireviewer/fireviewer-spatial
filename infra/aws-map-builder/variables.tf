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
  default     = 30

  validation {
    condition     = var.root_volume_size_gib == 30
    error_message = "The Amazon Linux 2023 root volume must remain fixed at 30 GiB."
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

variable "github_repository" {
  description = "GitHub owner/repository allowed to publish immutable Map Builder images."
  type        = string
  default     = "fireviewer/fireviewer-spatial"

  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "github_repository must use the owner/repository form."
  }
}

variable "github_oidc_provider_arn" {
  description = "Existing GitHub OIDC provider ARN. Leave null to create it in this stack."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.github_oidc_provider_arn == null || can(regex("^arn:[^:]+:iam::[0-9]{12}:oidc-provider/token\\.actions\\.githubusercontent\\.com$", var.github_oidc_provider_arn))
    error_message = "github_oidc_provider_arn must be the token.actions.githubusercontent.com provider ARN."
  }
}

variable "github_oidc_thumbprint" {
  description = "Root CA SHA-1 thumbprint used when Terraform creates the GitHub OIDC provider."
  type        = string
  default     = "6938fd4d98bab03faadb97b34396831e3780aea1"

  validation {
    condition     = can(regex("^[0-9a-f]{40}$", var.github_oidc_thumbprint))
    error_message = "github_oidc_thumbprint must be a lowercase SHA-1 hex digest."
  }
}

variable "batch_image_digest" {
  description = "Exact immutable Map Builder image digest used by the Batch job definition."
  type        = string
  default     = "sha256:d5a68c50865e0d895bdedad9854ff04cb6455105487234e20d8c5b6ec03f0f9d"

  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.batch_image_digest))
    error_message = "batch_image_digest must be a full sha256 digest."
  }
}

variable "hf_export_token_parameter_arn" {
  description = "Existing SSM SecureString ARN containing the Hugging Face token used only by the exporter job."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.hf_export_token_parameter_arn == null || can(regex("^arn:[^:]+:ssm:eu-west-3:[0-9]{12}:parameter/[A-Za-z0-9_.\\/-]+$", var.hf_export_token_parameter_arn))
    error_message = "hf_export_token_parameter_arn must be an eu-west-3 SSM parameter ARN."
  }
}

variable "hf_dataset_id" {
  description = "Public Hugging Face dataset receiving immutable tiled Map Builder runtimes."
  type        = string
  default     = "fireviewer/simple-measured-scenes-v1"

  validation {
    condition     = can(regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$", var.hf_dataset_id))
    error_message = "hf_dataset_id must use the owner/dataset form."
  }
}

variable "vercel_team_slug" {
  description = "Vercel team slug used by the Team OIDC issuer. Null keeps the backend AWS role disabled."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.vercel_team_slug == null || can(regex("^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$", var.vercel_team_slug))
    error_message = "vercel_team_slug must be null or the lowercase slug shown in the Vercel team URL."
  }
}

variable "vercel_project_name" {
  description = "Exact Vercel backend project allowed to assume the AWS map administration role."
  type        = string
  default     = "fireviewer-api"

  validation {
    condition     = var.vercel_project_name == "fireviewer-api"
    error_message = "Only the fireviewer-api Vercel project is authorized for Map Builder administration."
  }
}

variable "vercel_oidc_provider_arn" {
  description = "Existing Team-scoped Vercel OIDC provider ARN. Leave null to create it in this stack."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.vercel_oidc_provider_arn == null || can(regex("^arn:[^:]+:iam::[0-9]{12}:oidc-provider/oidc\\.vercel\\.com/[a-z0-9-]+$", var.vercel_oidc_provider_arn))
    error_message = "vercel_oidc_provider_arn must be a Team-scoped oidc.vercel.com provider ARN."
  }
}

variable "azure_oidc_tenant_id" {
  description = "Microsoft Entra tenant trusted for the Azure Container Apps backend. Null disables Azure federation."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.azure_oidc_tenant_id == null || can(regex("^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", var.azure_oidc_tenant_id))
    error_message = "azure_oidc_tenant_id must be null or a lowercase UUID."
  }
}

variable "azure_oidc_audience" {
  description = "Application ID URI requested by the Azure managed identity and accepted by AWS STS."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.azure_oidc_audience == null || can(regex("^(api|urn)://[A-Za-z0-9._:/-]{3,240}$", var.azure_oidc_audience))
    error_message = "azure_oidc_audience must be null or an api:// / urn:// application identifier URI."
  }
}

variable "azure_managed_identity_principal_id" {
  description = "Object (principal) ID of the exact Azure managed identity allowed to assume the backend role."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.azure_managed_identity_principal_id == null || can(regex("^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", var.azure_managed_identity_principal_id))
    error_message = "azure_managed_identity_principal_id must be null or a lowercase UUID."
  }
}

check "azure_oidc_configuration_is_complete" {
  assert {
    condition = length(compact([
      var.azure_oidc_tenant_id,
      var.azure_oidc_audience,
      var.azure_managed_identity_principal_id,
      ])) == 0 || length(compact([
      var.azure_oidc_tenant_id,
      var.azure_oidc_audience,
      var.azure_managed_identity_principal_id,
    ])) == 3
    error_message = "Azure OIDC federation requires tenant ID, audience and managed identity principal ID together."
  }
}

check "batch_requires_g2" {
  assert {
    condition     = !var.enable_batch || var.g2_validated
    error_message = "AWS Batch cannot be enabled before direct EC2 gates G2 and G3 pass."
  }
}

check "batch_requires_hf_export_secret" {
  assert {
    condition     = !local.batch_activation_requested || var.hf_export_token_parameter_arn != null
    error_message = "Enabled production Batch requires the existing Hugging Face token SSM parameter ARN."
  }
}
