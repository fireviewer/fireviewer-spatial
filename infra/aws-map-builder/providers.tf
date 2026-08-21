terraform {
  required_version = ">= 1.6.0"

  backend "s3" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = local.common_tags
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
  region     = data.aws_region.current.region

  name_slug = "${var.name_prefix}-${local.account_id}-${local.region}"

  common_tags = {
    Application = "FireViewer"
    Component   = "MapBuilder"
    ManagedBy   = "Terraform"
    Lifecycle   = "EphemeralCompute"
    Reference   = "map-builder-reference-v1"
  }
}
