data "aws_caller_identity" "current" {}

locals {
  # Who may assume the roles: this account (on the emulator, Moto's default)
  account_principal = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
}

# Training: the data pipeline (DVC) and the training jobs (MLflow)
resource "aws_iam_role" "training" {
  name        = "${var.name_prefix}-${var.environment}-training"
  description = "Runs the data pipeline and training: pushes dataset versions, logs model artifacts."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { AWS = local.account_principal }
    }]
  })
}

# Serving: the API only downloads @champion artifacts
resource "aws_iam_role" "serving" {
  name        = "${var.name_prefix}-${var.environment}-serving"
  description = "Runs the API: reads model artifacts, writes nothing."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { AWS = local.account_principal }
    }]
  })
}

# Least privilege for training: read and write objects in the two buckets.
# No deletes in the DVC remote (dataset versions are kept), no bucket or
# policy changes, nothing outside these buckets.
data "aws_iam_policy_document" "training" {
  statement {
    sid       = "ListBuckets"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [for b in aws_s3_bucket.this : b.arn]
  }

  statement {
    sid       = "PushDatasetVersions"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.this["dvc"].arn}/*"]
  }

  statement {
    sid       = "ReadWriteModelArtifacts"
    actions   = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.this["mlflow"].arn}/*"]
  }

  statement {
    sid       = "UseDataKey"
    actions   = ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"]
    resources = [aws_kms_key.data.arn]
  }
}

# Least privilege for serving: read model artifacts only
data "aws_iam_policy_document" "serving" {
  statement {
    sid       = "ListModelArtifacts"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.this["mlflow"].arn]
  }

  statement {
    sid       = "ReadModelArtifacts"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.this["mlflow"].arn}/*"]
  }

  statement {
    sid       = "DecryptWithDataKey"
    actions   = ["kms:Decrypt"]
    resources = [aws_kms_key.data.arn]
  }
}

resource "aws_iam_policy" "training" {
  name        = "${var.name_prefix}-${var.environment}-training-s3"
  description = "S3 access for the data pipeline and training, limited to the DVC and MLflow buckets."
  policy      = data.aws_iam_policy_document.training.json
}

resource "aws_iam_policy" "serving" {
  name        = "${var.name_prefix}-${var.environment}-serving-s3"
  description = "Read-only access to the MLflow artifact bucket for the API."
  policy      = data.aws_iam_policy_document.serving.json
}

resource "aws_iam_role_policy_attachment" "training" {
  role       = aws_iam_role.training.name
  policy_arn = aws_iam_policy.training.arn
}

resource "aws_iam_role_policy_attachment" "serving" {
  role       = aws_iam_role.serving.name
  policy_arn = aws_iam_policy.serving.arn
}
