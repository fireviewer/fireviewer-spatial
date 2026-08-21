locals {
  work_bucket_name   = "fireviewer-map-work-${local.account_id}-${local.region}"
  builds_bucket_name = "fireviewer-map-builds-${local.account_id}-${local.region}"
}

resource "aws_s3_bucket" "work" {
  bucket        = local.work_bucket_name
  force_destroy = false
}

resource "aws_s3_bucket" "builds" {
  bucket        = local.builds_bucket_name
  force_destroy = false
}

resource "aws_s3_bucket_ownership_controls" "work" {
  bucket = aws_s3_bucket.work.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_ownership_controls" "builds" {
  bucket = aws_s3_bucket.builds.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "work" {
  bucket                  = aws_s3_bucket.work.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "builds" {
  bucket                  = aws_s3_bucket.builds.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "work" {
  bucket = aws_s3_bucket.work.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "builds" {
  bucket = aws_s3_bucket.builds.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "builds" {
  bucket = aws_s3_bucket.builds.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "work" {
  bucket = aws_s3_bucket.work.id

  rule {
    id     = "failed-seven-days"
    status = "Enabled"

    filter {
      prefix = "failed/"
    }

    expiration {
      days = 7
    }
  }

  rule {
    id     = "cache-retention"
    status = "Enabled"

    filter {
      prefix = "cache/"
    }

    expiration {
      days = var.cache_retention_days
    }
  }

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "builds" {
  bucket = aws_s3_bucket.builds.id

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

data "aws_iam_policy_document" "work_bucket_tls" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions   = ["s3:*"]
    resources = [aws_s3_bucket.work.arn, "${aws_s3_bucket.work.arn}/*"]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

data "aws_iam_policy_document" "builds_bucket_tls" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions   = ["s3:*"]
    resources = [aws_s3_bucket.builds.arn, "${aws_s3_bucket.builds.arn}/*"]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "work_tls" {
  bucket = aws_s3_bucket.work.id
  policy = data.aws_iam_policy_document.work_bucket_tls.json
}

resource "aws_s3_bucket_policy" "builds_tls" {
  bucket = aws_s3_bucket.builds.id
  policy = data.aws_iam_policy_document.builds_bucket_tls.json
}
