locals {
  vercel_backend_oidc_enabled = var.vercel_team_slug != null
  vercel_oidc_issuer          = local.vercel_backend_oidc_enabled ? "https://oidc.vercel.com/${var.vercel_team_slug}" : null
  vercel_oidc_audience        = local.vercel_backend_oidc_enabled ? "https://vercel.com/${var.vercel_team_slug}" : null
  vercel_oidc_subject         = local.vercel_backend_oidc_enabled ? "owner:${var.vercel_team_slug}:project:${var.vercel_project_name}:environment:production" : null
  azure_backend_oidc_enabled  = var.azure_oidc_tenant_id != null && var.azure_oidc_audience != null && var.azure_managed_identity_principal_id != null
  map_admin_policy_enabled    = local.vercel_backend_oidc_enabled || local.azure_backend_oidc_enabled
}

resource "aws_iam_openid_connect_provider" "vercel" {
  count = local.vercel_backend_oidc_enabled && var.vercel_oidc_provider_arn == null ? 1 : 0

  url            = local.vercel_oidc_issuer
  client_id_list = [local.vercel_oidc_audience]
}

locals {
  vercel_oidc_provider_arn = !local.vercel_backend_oidc_enabled ? null : (
    var.vercel_oidc_provider_arn != null ? var.vercel_oidc_provider_arn : aws_iam_openid_connect_provider.vercel[0].arn
  )
}

data "aws_iam_policy_document" "vercel_backend_assume_role" {
  count = local.vercel_backend_oidc_enabled ? 1 : 0

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.vercel_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "oidc.vercel.com/${var.vercel_team_slug}:aud"
      values   = [local.vercel_oidc_audience]
    }

    condition {
      test     = "StringEquals"
      variable = "oidc.vercel.com/${var.vercel_team_slug}:sub"
      values   = [local.vercel_oidc_subject]
    }
  }
}

resource "aws_iam_role" "vercel_backend" {
  count = local.vercel_backend_oidc_enabled ? 1 : 0

  name                 = "${var.name_prefix}-vercel-backend"
  assume_role_policy   = data.aws_iam_policy_document.vercel_backend_assume_role[0].json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "map_admin" {
  count = local.map_admin_policy_enabled ? 1 : 0

  statement {
    sid     = "ListMapRequestObjects"
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = [
      aws_s3_bucket.work.arn,
    ]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["requests/*"]
    }
  }

  statement {
    sid     = "WriteImmutableMapRequests"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      "${aws_s3_bucket.work.arn}/requests/*",
    ]
  }

  statement {
    sid     = "ReadCompletedMapBuilds"
    effect  = "Allow"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.builds.arn}/maps/*",
    ]
  }

  # S3 deliberately returns AccessDenied instead of NoSuchKey when a caller
  # cannot list the bucket. The admin must distinguish an optional receipt
  # that has not been written yet from an authorization failure so it can
  # submit the exporter exactly once. Limit listing to validation artifacts.
  statement {
    sid       = "DetectMissingMapValidationArtifacts"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.builds.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values = [
        "maps/*/manifests/request.json",
        "maps/*/provenance/hf-viewer-publication.json",
        "maps/*/runtime/viewer-tiled/*",
        "maps/*/zone.done.json",
      ]
    }
  }

  statement {
    sid     = "SubmitOnlyFrozenMapBuilder"
    effect  = "Allow"
    actions = ["batch:SubmitJob"]
    resources = [
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job/*",
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job-queue/${var.name_prefix}",
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job-definition/${var.name_prefix}:*",
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job-definition/${var.name_prefix}-tile-shard:*",
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job-definition/${var.name_prefix}-hf-exporter:*",
    ]

    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/Application"
      values   = ["FireViewer"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/Component"
      values   = ["MapBuilder", "MapTileShard", "MapAssembler", "MapViewerExporter"]
    }
  }

  statement {
    sid     = "TagOnlySubmittedMapJobs"
    effect  = "Allow"
    actions = ["batch:TagResource"]
    resources = [
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job/*",
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job-queue/${var.name_prefix}",
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job-definition/${var.name_prefix}:*",
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job-definition/${var.name_prefix}-tile-shard:*",
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job-definition/${var.name_prefix}-hf-exporter:*",
    ]

    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/Application"
      values   = ["FireViewer"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/Component"
      values   = ["MapBuilder", "MapTileShard", "MapAssembler", "MapViewerExporter"]
    }
  }

  statement {
    sid       = "DescribeSubmittedJobs"
    effect    = "Allow"
    actions   = ["batch:DescribeJobs"]
    resources = ["*"]
  }

  statement {
    sid     = "TerminateOnlyFireViewerMapJobs"
    effect  = "Allow"
    actions = ["batch:TerminateJob"]
    resources = [
      "arn:${local.partition}:batch:${local.region}:${local.account_id}:job/*",
    ]

    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/Application"
      values   = ["FireViewer"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/Component"
      values   = ["MapBuilder", "MapTileShard", "MapAssembler", "MapViewerExporter"]
    }
  }
}

resource "aws_iam_role_policy" "vercel_backend" {
  count = local.vercel_backend_oidc_enabled ? 1 : 0

  name   = "${var.name_prefix}-vercel-backend"
  role   = aws_iam_role.vercel_backend[0].id
  policy = data.aws_iam_policy_document.map_admin[0].json
}

check "vercel_backend_requires_batch" {
  assert {
    condition     = !local.vercel_backend_oidc_enabled || local.batch_activation_requested
    error_message = "The Vercel backend role must not be activated before AWS Batch is explicitly enabled after G2/G3."
  }
}

resource "aws_iam_openid_connect_provider" "azure" {
  count = local.azure_backend_oidc_enabled ? 1 : 0

  url            = "https://sts.windows.net/${var.azure_oidc_tenant_id}/"
  client_id_list = [var.azure_oidc_audience]
}

data "aws_iam_policy_document" "azure_backend_assume_role" {
  count = local.azure_backend_oidc_enabled ? 1 : 0

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.azure[0].arn]
    }

    condition {
      test     = "StringEquals"
      variable = "sts.windows.net/${var.azure_oidc_tenant_id}/:aud"
      values   = [var.azure_oidc_audience]
    }

    condition {
      test     = "StringEquals"
      variable = "sts.windows.net/${var.azure_oidc_tenant_id}/:sub"
      values   = [var.azure_managed_identity_principal_id]
    }
  }
}

resource "aws_iam_role" "azure_backend" {
  count = local.azure_backend_oidc_enabled ? 1 : 0

  name                 = "${var.name_prefix}-azure-backend"
  assume_role_policy   = data.aws_iam_policy_document.azure_backend_assume_role[0].json
  max_session_duration = 3600
}

resource "aws_iam_role_policy" "azure_backend" {
  count = local.azure_backend_oidc_enabled ? 1 : 0

  name   = "${var.name_prefix}-azure-backend"
  role   = aws_iam_role.azure_backend[0].id
  policy = data.aws_iam_policy_document.map_admin[0].json
}

check "azure_backend_requires_batch" {
  assert {
    condition     = !local.azure_backend_oidc_enabled || local.batch_activation_requested
    error_message = "The Azure backend role must not be activated before AWS Batch is explicitly enabled after G2/G3."
  }
}
