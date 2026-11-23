terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # Local state for the emulator. For a real account, use a remote backend
  # with locking, e.g.:
  #   backend "s3" { bucket = "<state bucket>" key = "claim-triage/terraform.tfstate"
  #                  region = "ca-central-1" use_lockfile = true }
}
