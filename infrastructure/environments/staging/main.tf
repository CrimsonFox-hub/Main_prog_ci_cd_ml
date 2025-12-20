# Сеть
resource "yandex_vpc_network" "mlops_network" {
  name = "${var.project_name}-network-${var.environment}"
}

resource "yandex_vpc_subnet" "mlops_subnet" {
  name           = "${var.project_name}-subnet-${var.environment}"
  zone           = var.zone
  network_id     = yandex_vpc_network.mlops_network.id
  v4_cidr_blocks = ["10.10.0.0/24"]
}

# Security Groups
resource "yandex_vpc_security_group" "k8s_cluster_sg" {
  name        = "${var.project_name}-k8s-cluster-sg-${var.environment}"
  network_id  = yandex_vpc_network.mlops_network.id
  description = "Security group for Kubernetes cluster"

  ingress {
    protocol       = "TCP"
    description    = "Kubernetes API"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 6443
  }

  ingress {
    protocol       = "TCP"
    description    = "HTTPS"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 443
  }

  ingress {
    protocol       = "TCP"
    description    = "HTTP"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 80
  }

  egress {
    protocol       = "ANY"
    description    = "Outgoing traffic"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "yandex_vpc_security_group" "k8s_nodes_sg" {
  name        = "${var.project_name}-k8s-nodes-sg-${var.environment}"
  network_id  = yandex_vpc_network.mlops_network.id
  description = "Security group for Kubernetes nodes"

  ingress {
    protocol          = "ANY"
    description       = "Communication between nodes"
    predefined_target = "self_security_group"
  }

  ingress {
    protocol       = "TCP"
    description    = "NodePort services"
    from_port      = 30000
    to_port        = 32767
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol       = "TCP"
    description    = "SSH"
    port           = 22
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol       = "ANY"
    description    = "Outgoing traffic"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

# Kubernetes Cluster
resource "yandex_kubernetes_cluster" "mlops_cluster" {
  name        = "${var.project_name}-cluster-${var.environment}"
  description = "Kubernetes cluster for MLOps project"
  network_id  = yandex_vpc_network.mlops_network.id

  cluster_ipv4_range = "10.20.0.0/16"
  service_ipv4_range = "10.21.0.0/16"
  node_ipv4_cidr_mask_size = 24

  master {
    version = "1.25"  # Используйте актуальную версию
    public_ip = true

    maintenance_policy {
      auto_upgrade = true
    }

    zonal {
      zone      = var.zone
      subnet_id = yandex_vpc_subnet.mlops_subnet.id
    }

    security_group_ids = [
      yandex_vpc_security_group.k8s_cluster_sg.id
    ]
  }

  service_account_id      = yandex_iam_service_account.k8s_sa.id
  node_service_account_id = yandex_iam_service_account.k8s_sa.id

  kms_provider {
    key_id = yandex_kms_symmetric_key.k8s_key.id
  }

  release_channel = "RAPID"
}

# Node Groups
resource "yandex_kubernetes_node_group" "cpu_pool" {
  cluster_id = yandex_kubernetes_cluster.mlops_cluster.id
  name       = "${var.project_name}-cpu-pool-${var.environment}"
  
  instance_template {
    platform_id = "standard-v3"
    
    network_interface {
      nat                = true
      subnet_ids         = [yandex_vpc_subnet.mlops_subnet.id]
      security_group_ids = [yandex_vpc_security_group.k8s_nodes_sg.id]
    }

    resources {
      memory = 8
      cores  = 4
    }

    boot_disk {
      type = "network-ssd"
      size = 64
    }

    scheduling_policy {
      preemptible = false
    }

    container_runtime {
      type = "containerd"
    }
  }

  scale_policy {
    auto_scale {
      min     = 2
      max     = 5
      initial = 2
    }
  }

  allocation_policy {
    location {
      zone = var.zone
    }
  }

  maintenance_policy {
    auto_upgrade = true
    auto_repair  = true
  }
}

# GPU Node Group (опционально)
resource "yandex_kubernetes_node_group" "gpu_pool" {
  count = var.environment == "production" ? 1 : 0
  
  cluster_id = yandex_kubernetes_cluster.mlops_cluster.id
  name       = "${var.project_name}-gpu-pool-${var.environment}"
  
  instance_template {
    platform_id = "standard-v3"
    
    resources {
      memory = 32
      cores  = 8
      gpus   = 1
    }

    network_interface {
      nat                = true
      subnet_ids         = [yandex_vpc_subnet.mlops_subnet.id]
      security_group_ids = [yandex_vpc_security_group.k8s_nodes_sg.id]
    }

    boot_disk {
      type = "network-ssd"
      size = 128
    }

    scheduling_policy {
      preemptible = false
    }
  }

  scale_policy {
    fixed_scale {
      size = 1
    }
  }

  allocation_policy {
    location {
      zone = var.zone
    }
  }
}

# Container Registry
resource "yandex_container_registry" "mlops_registry" {
  name = "${var.project_name}-registry"
}

# KMS ключ для шифрования секретов
resource "yandex_kms_symmetric_key" "k8s_key" {
  name              = "${var.project_name}-kms-key"
  description       = "KMS key for Kubernetes secrets"
  default_algorithm = "AES_256"
  rotation_period   = "8760h"  # 1 год
}

# Object Storage для Terraform state
resource "yandex_storage_bucket" "terraform_state" {
  bucket = "mlops-terraform-state"
  acl    = "private"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    enabled = true
    expiration {
      days = 365
    }
    noncurrent_version_expiration {
      days = 30
    }
  }
}

# Object Storage для DVC
resource "yandex_storage_bucket" "dvc_storage" {
  bucket = "${var.project_name}-dvc-storage"
  acl    = "private"

  versioning {
    enabled = true
  }
}

# Managed PostgreSQL
resource "yandex_mdb_postgresql_cluster" "mlops_db" {
  name        = "${var.project_name}-db-${var.environment}"
  environment = "PRODUCTION"
  network_id  = yandex_vpc_network.mlops_network.id

  config {
    version = 15
    resources {
      resource_preset_id = "s2.micro"
      disk_type_id       = "network-ssd"
      disk_size          = 20
    }

    postgresql_config = {
      max_connections                   = 100
      enable_parallel_hash              = true
      vacuum_cleanup_index_scale_factor = 0.2
      autovacuum_vacuum_scale_factor    = 0.34
      default_transaction_isolation     = "TRANSACTION_READ_COMMITTED"
      shared_preload_libraries          = "pg_stat_statements"
    }
  }

  host {
    zone      = var.zone
    subnet_id = yandex_vpc_subnet.mlops_subnet.id
  }

  database {
    name  = "credit_scoring"
    owner = "mlops_user"
  }

  user {
    name     = "mlops_user"
    password = random_password.db_password.result
    permission {
      database_name = "credit_scoring"
    }
  }
}

# Генерация пароля для БД
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Outputs
output "k8s_endpoint" {
  value = yandex_kubernetes_cluster.mlops_cluster.master.0.external_v4_endpoint
}

output "cluster_id" {
  value = yandex_kubernetes_cluster.mlops_cluster.id
}

output "registry_id" {
  value = yandex_container_registry.mlops_registry.id
}

output "database_host" {
  value = yandex_mdb_postgresql_cluster.mlops_db.host.0.fqdn
}

output "database_password" {
  value     = random_password.db_password.result
  sensitive = true
}