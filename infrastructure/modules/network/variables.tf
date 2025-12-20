variable "vpc_name" {
  type        = string
  description = "Name of the VPC"
}

variable "description" {
  type        = string
  description = "Description of the VPC"
  default     = "VPC for MLOps project"
}

variable "labels" {
  type        = map(string)
  description = "Labels for resources"
  default     = {}
}

variable "subnets" {
  type = list(object({
    name                 = string
    description          = optional(string, "")
    zone                 = string
    v4_cidr_blocks       = list(string)
    route_table_id       = optional(string)
    domain_name          = optional(string)
    domain_name_servers  = optional(list(string), ["8.8.8.8", "8.8.4.4"])
    labels               = optional(map(string), {})
  }))
  description = "List of subnets to create"
  default     = []
}

# Security Groups CIDR blocks
variable "allowed_internal_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed for internal communication"
  default     = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
}

variable "allowed_k8s_api_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed for Kubernetes API access"
  default     = ["0.0.0.0/0"]
}

variable "allowed_ssh_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed for SSH access"
  default     = ["0.0.0.0/0"]
}

variable "allowed_http_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed for HTTP/HTTPS access"
  default     = ["0.0.0.0/0"]
}

variable "allowed_nodeport_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed for NodePort access"
  default     = ["0.0.0.0/0"]
}

variable "allowed_database_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed for database access"
  default     = ["10.0.0.0/8"]
}

variable "additional_ingress_rules" {
  type = list(object({
    protocol       = string
    description    = string
    v4_cidr_blocks = list(string)
    port           = optional(number)
    from_port      = optional(number)
    to_port        = optional(number)
  }))
  description = "Additional ingress rules"
  default     = []
}

variable "additional_egress_rules" {
  type = list(object({
    protocol       = string
    description    = string
    v4_cidr_blocks = list(string)
    port           = optional(number)
    from_port      = optional(number)
    to_port        = optional(number)
  }))
  description = "Additional egress rules"
  default     = []
}