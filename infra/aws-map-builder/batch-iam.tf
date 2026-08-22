data "aws_iam_policy_document" "batch_service_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_service" {
  count = local.batch_activation_requested ? 1 : 0

  name               = "${var.name_prefix}-batch-service"
  assume_role_policy = data.aws_iam_policy_document.batch_service_assume_role.json
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  count = local.batch_activation_requested ? 1 : 0

  role       = aws_iam_role.batch_service[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role" "batch_instance" {
  count = local.batch_activation_requested ? 1 : 0

  name               = "${var.name_prefix}-batch-instance"
  assume_role_policy = data.aws_iam_policy_document.worker_assume_role.json
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  count = local.batch_activation_requested ? 1 : 0

  role       = aws_iam_role.batch_instance[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_instance" {
  count = local.batch_activation_requested ? 1 : 0

  name = "${var.name_prefix}-batch-instance"
  role = aws_iam_role.batch_instance[0].name
}

data "aws_iam_policy_document" "ecs_tasks_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_execution" {
  count = local.batch_activation_requested ? 1 : 0

  name               = "${var.name_prefix}-batch-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

resource "aws_iam_role_policy_attachment" "batch_execution" {
  count = local.batch_activation_requested ? 1 : 0

  role       = aws_iam_role.batch_execution[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "batch_job" {
  count = local.batch_activation_requested ? 1 : 0

  name               = "${var.name_prefix}-batch-job"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

data "aws_iam_policy_document" "batch_job" {
  statement {
    sid     = "ListJobObjects"
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = [
      aws_s3_bucket.work.arn,
      aws_s3_bucket.builds.arn,
    ]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values = [
        "requests/*",
        "cache/*",
        "maps/*",
      ]
    }
  }

  statement {
    sid    = "ReadJobInputs"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]
    resources = [
      "${aws_s3_bucket.work.arn}/requests/*",
      "${aws_s3_bucket.work.arn}/cache/*",
    ]
  }

  statement {
    sid    = "WriteCurrentBuild"
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:ListMultipartUploadParts",
      "s3:PutObject",
    ]
    resources = ["${aws_s3_bucket.builds.arn}/maps/*"]
  }

  statement {
    sid    = "WriteOnlyCurrentTileShards"
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
      "s3:PutObject",
    ]
    resources = ["${aws_s3_bucket.work.arn}/requests/*/shards/*"]
  }
}

resource "aws_iam_role_policy" "batch_job" {
  count = local.batch_activation_requested ? 1 : 0

  name   = "${var.name_prefix}-current-job"
  role   = aws_iam_role.batch_job[0].id
  policy = data.aws_iam_policy_document.batch_job.json
}

resource "aws_iam_role" "batch_hf_execution" {
  count = local.batch_activation_requested ? 1 : 0

  name               = "${var.name_prefix}-hf-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

resource "aws_iam_role_policy_attachment" "batch_hf_execution" {
  count = local.batch_activation_requested ? 1 : 0

  role       = aws_iam_role.batch_hf_execution[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "batch_hf_execution_secret" {
  count = local.batch_activation_requested ? 1 : 0

  statement {
    sid       = "ReadOnlyHuggingFaceToken"
    effect    = "Allow"
    actions   = ["ssm:GetParameters"]
    resources = [var.hf_export_token_parameter_arn]
  }
}

resource "aws_iam_role_policy" "batch_hf_execution_secret" {
  count = local.batch_activation_requested ? 1 : 0

  name   = "${var.name_prefix}-hf-token"
  role   = aws_iam_role.batch_hf_execution[0].id
  policy = data.aws_iam_policy_document.batch_hf_execution_secret[0].json
}

resource "aws_iam_role" "batch_hf_job" {
  count = local.batch_activation_requested ? 1 : 0

  name               = "${var.name_prefix}-hf-job"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

data "aws_iam_policy_document" "batch_hf_job" {
  statement {
    sid     = "ListMapBuildObjects"
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = [
      aws_s3_bucket.builds.arn,
    ]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["maps/*"]
    }
  }

  statement {
    sid    = "ReadValidatedMapViewer"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]
    resources = ["${aws_s3_bucket.builds.arn}/maps/*"]
  }

  statement {
    sid     = "WriteOnlyHuggingFaceReceipt"
    effect  = "Allow"
    actions = ["s3:PutObject"]
    resources = [
      "${aws_s3_bucket.builds.arn}/maps/*/provenance/hf-viewer-publication.json",
    ]
  }
}

resource "aws_iam_role_policy" "batch_hf_job" {
  count = local.batch_activation_requested ? 1 : 0

  name   = "${var.name_prefix}-hf-publication"
  role   = aws_iam_role.batch_hf_job[0].id
  policy = data.aws_iam_policy_document.batch_hf_job.json
}
