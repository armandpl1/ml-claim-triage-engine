variable "region" {
  description = "AWS region."
  type        = string
  default     = "ca-central-1"
}

variable "environment" {
  description = "Deployment environment, used in names and tags."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be dev or prod."
  }
}

variable "name_prefix" {
  description = "Prefix for every resource name."
  type        = string
  default     = "claim-triage"
}

variable "use_emulator" {
  description = "Point the AWS provider at the local emulator (Moto) instead of AWS."
  type        = bool
  default     = true
}

variable "emulator_endpoint" {
  description = "Emulator URL (infra/docker-compose.yml)."
  type        = string
  default     = "http://localhost:15100"
}

variable "noncurrent_days" {
  description = "Days to keep overwritten or deleted object versions before expiring them."
  type        = number
  default     = 90
}
