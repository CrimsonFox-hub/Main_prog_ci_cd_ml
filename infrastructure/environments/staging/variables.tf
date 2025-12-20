variable "yc_token" {
  type        = string
  description = "Yandex Cloud OAuth token"
  sensitive   = true
}

variable "cloud_id" {
  type        = string
  description = "Yandex Cloud ID"
  sensitive   = true
}

variable "folder_id" {
  type        = string
  description = "Yandex Cloud folder ID"
  sensitive   = true
}

variable "zone" {
  type        = string
  description = "YC zone"
  default     = "ru-central1-a"
}

variable "k8s_token" {
  type        = string
  description = "Kubernetes service account token"
  sensitive   = true
}

variable "s3_access_key" {
  type        = string
  description = "S3 access key for Terraform state"
  sensitive   = true
}

variable "s3_secret_key" {
  type        = string
  description = "S3 secret key for Terraform state"
  sensitive   = true
}

variable "project_name" {
  type        = string
  description = "Project name"
  default     = "credit-scoring-mlops"
}

variable "environment" {
  type        = string
  description = "Environment name"
  default     = "staging"
}