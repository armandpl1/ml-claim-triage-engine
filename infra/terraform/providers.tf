provider "aws" {
  region = var.region

  # Against the emulator: fake credentials, no account lookups, path-style S3
  access_key                  = var.use_emulator ? "test" : null
  secret_key                  = var.use_emulator ? "test" : null
  skip_credentials_validation = var.use_emulator
  skip_requesting_account_id  = var.use_emulator
  skip_metadata_api_check     = var.use_emulator
  s3_use_path_style           = var.use_emulator

  dynamic "endpoints" {
    for_each = var.use_emulator ? [var.emulator_endpoint] : []
    content {
      s3  = endpoints.value
      iam = endpoints.value
      sts = endpoints.value
      kms = endpoints.value
    }
  }

  default_tags {
    tags = {
      project     = "ml-claim-triage-engine"
      environment = var.environment
      managed_by  = "terraform"
    }
  }
}
