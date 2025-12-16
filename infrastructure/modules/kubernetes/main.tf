# Модуль для создания управляемого Kubernetes кластера
terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.89"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.9"
    }
  }
}

# Локальные переменные
locals {
  cluster_name = "${var.project_name}-${var.environment}"
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Component   = "Kubernetes"
  }
}

# Создание сервисного аккаунта для кластера
resource "yandex_iam_service_account" "cluster" {
  name        = "${local.cluster_name}-cluster-sa"
  description = "Service account for Kubernetes cluster"
  folder_id   = var.folder_id
}

# Назначение ролей сервисному аккаунту
resource "yandex_resourcemanager_folder_iam_member" "cluster_roles" {
  for_each = toset([
    "editor",
    "container-registry.images.puller",
    "monitoring.editor"
  ])
  
  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.cluster.id}"
}

# Создание сервисного аккаунта для нод
resource "yandex_iam_service_account" "node" {
  name        = "${local.cluster_name}-node-sa"
  description = "Service account for Kubernetes nodes"
  folder_id   = var.folder_id
}

# Назначение ролей сервисному аккаунту нод
resource "yandex_resourcemanager_folder_iam_member" "node_roles" {
  for_each = toset([
    "container-registry.images.puller",
    "monitoring.editor",
    "logging.writer",
    "kms.keys.encrypterDecrypter"
  ])
  
  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.node.id}"
}

# Создание KMS ключа для шифрования секретов
resource "yandex_kms_symmetric_key" "kms_key" {
  name              = "${local.cluster_name}-kms-key"
  description       = "KMS key for Kubernetes secrets encryption"
  default_algorithm = "AES_256"
  rotation_period   = "8760h" # 1 год
  
  labels = local.common_tags
}

# Создание управляемого Kubernetes кластера
resource "yandex_kubernetes_cluster" "primary" {
  name       = local.cluster_name
  network_id = var.network_id
  folder_id  = var.folder_id
  
  cluster_ipv4_range = var.cluster_ipv4_range
  service_ipv4_range = var.service_ipv4_range
  node_ipv4_cidr_mask_size = var.node_ipv4_cidr_mask_size
  
  # Конфигурация мастера
  master {
    version   = var.kubernetes_version
    public_ip = true
    
    master_location {
      zone      = var.zone
      subnet_id = var.subnet_id
    }
    
    security_group_ids = var.security_group_ids
    
    # Настройки мастера
    maintenance_policy {
      auto_upgrade = true
      
      maintenance_window {
        day        = "saturday"
        start_time = "23:00"
        duration   = "3h"
      }
    }
  }
  
  # Шифрование секретов с KMS
  kms_provider {
    key_id = yandex_kms_symmetric_key.kms_key.id
  }
  
  # Сервисные аккаунты
  service_account_id      = yandex_iam_service_account.cluster.id
  node_service_account_id = yandex_iam_service_account.node.id
  
  # Настройки сети
  network_policy_provider = "CALICO"
  
  # Настройки мониторинга и логирования
  cluster_logging_enabled    = true
  cluster_logging_kafka_id   = var.kafka_id
  cluster_monitoring_enabled = true
  
  labels = local.common_tags
}

# Создание групп нод
resource "yandex_kubernetes_node_group" "cpu_nodes" {
  cluster_id  = yandex_kubernetes_cluster.primary.id
  name        = "${local.cluster_name}-cpu-nodes"
  description = "CPU-optimized node group for general workloads"
  
  instance_template {
    platform_id = "standard-v3"
    
    resources {
      memory = var.cpu_node_memory
      cores  = var.cpu_node_cores
    }
    
    boot_disk {
      type = "network-ssd"
      size = var.cpu_node_disk_size
    }
    
    scheduling_policy {
      preemptible = var.environment != "production"
    }
    
    network_interface {
      subnet_ids = [var.subnet_id]
      nat        = true
    }
    
    container_runtime {
      type = "containerd"
    }
    
    # Настройки безопасности
    security_group_ids = var.security_group_ids
    
    labels = {
      "node-pool" = "cpu"
      "environment" = var.environment
    }
    
    metadata = {
      ssh-keys = "ubuntu:${file(var.ssh_public_key_path)}"
    }
  }
  
  # Политика масштабирования
  scale_policy {
    auto_scale {
      min     = var.cpu_node_min_count
      max     = var.cpu_node_max_count
      initial = var.cpu_node_initial_count
    }
  }
  
  # Политика обновления
  update_policy {
    max_unavailable       = 1
    max_expansion         = 0
    max_creation_time     = "30m"
    node_startup_timeout  = "15m"
  }
  
  # Allocation policy
  allocation_policy {
    location {
      zone = var.zone
    }
  }
  
  # Maintenance policy
  maintenance_policy {
    auto_upgrade = true
    auto_repair  = true
    
    maintenance_window {
      day        = "sunday"
      start_time = "02:00"
      duration   = "4h"
    }
  }
  
  labels = local.common_tags
}

# Создание GPU группы нод (опционально)
resource "yandex_kubernetes_node_group" "gpu_nodes" {
  count = var.enable_gpu_nodes ? 1 : 0
  
  cluster_id  = yandex_kubernetes_cluster.primary.id
  name        = "${local.cluster_name}-gpu-nodes"
  description = "GPU-optimized node group for ML workloads"
  
  instance_template {
    platform_id = "gpu-standard-v3"
    gpu_environment = "runc-docker"
    
    resources {
      memory = var.gpu_node_memory
      cores  = var.gpu_node_cores
      gpus   = var.gpu_node_gpus
    }
    
    boot_disk {
      type = "network-ssd"
      size = var.gpu_node_disk_size
    }
    
    scheduling_policy {
      preemptible = false # GPU instances обычно не preemptible
    }
    
    network_interface {
      subnet_ids = [var.subnet_id]
      nat        = true
    }
    
    container_runtime {
      type = "containerd"
    }
    
    # Настройки безопасности
    security_group_ids = var.security_group_ids
    
    labels = {
      "node-pool" = "gpu"
      "gpu-type"  = var.gpu_type
      "environment" = var.environment
    }
    
    taints = [{
      key    = "sku"
      value  = "gpu"
      effect = "NO_SCHEDULE"
    }]
  }
  
  scale_policy {
    fixed_scale {
      size = var.gpu_node_count
    }
  }
  
  allocation_policy {
    location {
      zone = var.zone
    }
  }
  
  labels = local.common_tags
}

# Настройка провайдера Kubernetes
provider "kubernetes" {
  host  = yandex_kubernetes_cluster.primary.master[0].external_v4_endpoint
  token = data.yandex_client_config.client.iam_token
  
  cluster_ca_certificate = base64decode(
    yandex_kubernetes_cluster.primary.master[0].cluster_ca_certificate
  )
}

# Настройка провайдера Helm
provider "helm" {
  kubernetes {
    host  = yandex_kubernetes_cluster.primary.master[0].external_v4_endpoint
    token = data.yandex_client_config.client.iam_token
    
    cluster_ca_certificate = base64decode(
      yandex_kubernetes_cluster.primary.master[0].cluster_ca_certificate
    )
  }
}

# Создание namespace для приложения
resource "kubernetes_namespace" "app_namespace" {
  metadata {
    name = var.app_namespace
    labels = {
      name = var.app_namespace
    }
  }
}

# Сетевые политики
resource "kubernetes_network_policy" "default_deny" {
  metadata {
    name      = "default-deny-all"
    namespace = kubernetes_namespace.app_namespace.metadata[0].name
  }
  
  spec {
    pod_selector {}
    policy_types = ["Ingress", "Egress"]
  }
}

resource "kubernetes_network_policy" "allow_internal" {
  metadata {
    name      = "allow-internal"
    namespace = kubernetes_namespace.app_namespace.metadata[0].name
  }
  
  spec {
    pod_selector {}
    
    ingress {
      from {
        pod_selector {}
      }
    }
    
    egress {
      to {
        pod_selector {}
      }
    }
    
    policy_types = ["Ingress", "Egress"]
  }
}

# Конфигурационный Map для приложения
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "app-config"
    namespace = kubernetes_namespace.app_namespace.metadata[0].name
  }
  
  data = {
    "app_config.yaml" = yamlencode({
      environment = var.environment
      logging = {
        level = var.log_level
      }
      monitoring = {
        enabled = true
        port    = 9090
      }
      database = {
        host = var.db_host
        port = var.db_port
      }
    })
  }
}

# Output values
output "cluster_id" {
  description = "ID of the created Kubernetes cluster"
  value       = yandex_kubernetes_cluster.primary.id
}

output "cluster_external_endpoint" {
  description = "External endpoint of the Kubernetes cluster"
  value       = yandex_kubernetes_cluster.primary.master[0].external_v4_endpoint
}

output "cluster_ca_certificate" {
  description = "CA certificate of the Kubernetes cluster"
  value       = yandex_kubernetes_cluster.primary.master[0].cluster_ca_certificate
  sensitive   = true
}

output "kubeconfig" {
  description = "Kubeconfig for accessing the cluster"
  value = yamldecode(<<-EOT
    apiVersion: v1
    clusters:
    - cluster:
        certificate-authority-data: ${yandex_kubernetes_cluster.primary.master[0].cluster_ca_certificate}
        server: ${yandex_kubernetes_cluster.primary.master[0].external_v4_endpoint}
      name: ${local.cluster_name}
    contexts:
    - context:
        cluster: ${local.cluster_name}
        user: ${local.cluster_name}-admin
      name: ${local.cluster_name}
    current-context: ${local.cluster_name}
    kind: Config
    preferences: {}
    users:
    - name: ${local.cluster_name}-admin
      user:
        token: ${data.yandex_client_config.client.iam_token}
  EOT
  )
  sensitive = true
}

output "node_groups" {
  description = "Created node groups"
  value = {
    cpu = {
      name = yandex_kubernetes_node_group.cpu_nodes.name
      size = var.cpu_node_initial_count
    }
    gpu = var.enable_gpu_nodes ? {
      name = yandex_kubernetes_node_group.gpu_nodes[0].name
      size = var.gpu_node_count
    } : null
  }
}