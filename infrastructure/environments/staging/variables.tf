# Input variables for staging environment

variable "yc_token" {
  type        = string
  description = "Yandex Cloud OAuth token or IAM token"
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
  description = "Default zone for resources"
  default     = "ru-central1-a"
}

variable "region" {
  type        = string
  description = "Default region for resources"
  default     = "ru-central1"
}

# S3 credentials for Terraform backend
variable "s3_access_key" {
  type        = string
  description = "S3 access key for Terraform state"
  sensitive   = true
  default     = null
}

variable "s3_secret_key" {
  type        = string
  description = "S3 secret key for Terraform state"
  sensitive   = true
  default     = null
}

# Optional: Pre-defined service account ID
variable "service_account_id" {
  type        = string
  description = "Service account ID for cluster (optional)"
  default     = null
}

# Optional: Existing KMS key ID
variable "kms_key_id" {
  type        = string
  description = "Existing KMS key ID for cluster secrets (optional)"
  default     = null
}

# Environment specific variables
variable "node_count" {
  type        = number
  description = "Number of worker nodes in Kubernetes cluster"
  default     = 2
}

variable "enable_gpu_nodes" {
  type        = bool
  description = "Enable GPU node pool for ML training"
  default     = false
}

variable "enable_monitoring_stack" {
  type        = bool
  description = "Enable monitoring stack (Prometheus, Grafana, Loki)"
  default     = true
}

variable "enable_logging" {
  type        = bool
  description = "Enable cluster logging"
  default     = true
}

variable "domain_name" {
  type        = string
  description = "Domain name for applications (optional)"
  default     = null
}

variable "ssl_certificate_id" {
  type        = string
  description = "SSL certificate ID for load balancer (optional)"
  default     = null
}

# Network CIDRs
variable "vpc_cidr" {
  type        = string
  description = "VPC CIDR block"
  default     = "10.10.0.0/16"
}

variable "subnet_cidr" {
  type        = string
  description = "Subnet CIDR block"
  default     = "10.10.0.0/24"
}

variable "k8s_pod_cidr" {
  type        = string
  description = "Kubernetes pod CIDR"
  default     = "10.20.0.0/16"
}

variable "k8s_service_cidr" {
  type        = string
  description = "Kubernetes service CIDR"
  default     = "10.21.0.0/16"
}