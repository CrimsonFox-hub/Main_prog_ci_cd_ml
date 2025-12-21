variable "cluster_name" {
  description = "Имя кластера Kubernetes"
  type        = string
  default     = "credit-scoring-cluster"
}

variable "k8s_version" {
  description = "Версия Kubernetes"
  type        = string
  default     = "1.27"
}

variable "node_groups" {
  description = "Конфигурация node groups"
  type = map(object({
    instance_type = string
    cores         = number
    memory        = number
    disk_size     = number
    min_size      = number
    max_size      = number
    gpus          = optional(number)
    node_labels   = optional(map(string))
  }))
  default = {
    cpu = {
      instance_type = "standard-v2"
      cores         = 4
      memory        = 8
      disk_size     = 50
      min_size      = 2
      max_size      = 5
      node_labels   = { "node-pool" = "cpu" }
    }
    gpu = {
      instance_type = "gpu-standard-v2-t4"
      cores         = 8
      memory        = 32
      disk_size     = 100
      gpus          = 1
      min_size      = 0
      max_size      = 2
      node_labels   = { "node-pool" = "gpu", "gpu-type" = "nvidia-t4" }
    }
  }
}

variable "region" {
  description = "Регион Yandex Cloud"
  type        = string
  default     = "ru-central1"
}

variable "zones" {
  description = "Зоны доступности"
  type        = list(string)
  default     = ["ru-central1-a", "ru-central1-b"]
}