output "dvc_bucket" {
  description = "DVC remote: versions of the datasets."
  value       = aws_s3_bucket.this["dvc"].id
}

output "mlflow_bucket" {
  description = "MLflow artifact store: models, plots, tokenizer."
  value       = aws_s3_bucket.this["mlflow"].id
}

output "logs_bucket" {
  description = "Bucket receiving the S3 access logs of the data buckets."
  value       = aws_s3_bucket.logs.id
}

output "training_role_arn" {
  description = "Role the data pipeline and training jobs assume."
  value       = aws_iam_role.training.arn
}

output "serving_role_arn" {
  description = "Role the API assumes (read-only on model artifacts)."
  value       = aws_iam_role.serving.arn
}

output "data_key_arn" {
  description = "Customer-managed KMS key encrypting every bucket."
  value       = aws_kms_key.data.arn
}
