# Customer-managed key for every bucket: rotation on, and access controlled
# by IAM (the training and serving roles may use it, nothing else is granted)
resource "aws_kms_key" "data" {
  description             = "Encrypts the claim-triage data, model artifact and log buckets."
  enable_key_rotation     = true
  deletion_window_in_days = 7
}

resource "aws_kms_alias" "data" {
  name          = "alias/${var.name_prefix}-${var.environment}-data"
  target_key_id = aws_kms_key.data.key_id
}
