locals {
  buckets = {
    dvc    = "${var.name_prefix}-${var.environment}-dvc"    # DVC remote: versions of the datasets
    mlflow = "${var.name_prefix}-${var.environment}-mlflow" # MLflow artifacts: models, plots, tokenizer
  }
}

resource "aws_s3_bucket" "this" {
  for_each = local.buckets
  bucket   = each.value
}

# Access logs of the data buckets. Not logged itself: that would loop.
#trivy:ignore:AVD-AWS-0089
resource "aws_s3_bucket" "logs" {
  bucket = "${var.name_prefix}-${var.environment}-access-logs"
}

# S3 delivers access logs only to a bucket that lets the logging service
# write to it: allowed for this account's two data buckets, nothing else
data "aws_iam_policy_document" "logs" {
  statement {
    sid       = "S3ServerAccessLogs"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.logs.arn}/*"]

    principals {
      type        = "Service"
      identifiers = ["logging.s3.amazonaws.com"]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = [for b in aws_s3_bucket.this : b.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_s3_bucket_policy" "logs" {
  bucket = aws_s3_bucket.logs.id
  policy = data.aws_iam_policy_document.logs.json

  depends_on = [aws_s3_bucket_public_access_block.logs]
}

resource "aws_s3_bucket_logging" "this" {
  for_each      = aws_s3_bucket.this
  bucket        = each.value.id
  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "${each.key}/"

  depends_on = [aws_s3_bucket_policy.logs]
}

# Keep every version: a bad overwrite or delete of a dataset or model can
# be undone
resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
    bucket_key_enabled = true # fewer KMS calls, lower cost
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# DVC and MLflow write content-addressed or run-scoped keys and rarely
# overwrite, so old versions are only kept long enough to recover a mistake
resource "aws_s3_bucket_lifecycle_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_days
    }
  }

  depends_on = [aws_s3_bucket_versioning.this]
}

# The log bucket gets the same protections, declared on their own so
# scanners can link each setting to the bucket
resource "aws_s3_bucket_versioning" "logs" {
  bucket = aws_s3_bucket.logs.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
