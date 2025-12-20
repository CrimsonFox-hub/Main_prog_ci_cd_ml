# Staging environment for MLOps Credit Scoring

# Local variables
# Local variables
locals {
  project_name    = "credit-scoring"
  environment     = "production"
  region          = "ru-central1"
  zones           = ["ru-central1-a", "ru-central1-b", "ru-central1-c"]  # Multi-zone для отказоустойчивости
  
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "Terraform"
    Repository  = "github.com/CrimsonFox-hub/Main_prog_ci_cd_ml.git"
    Tier        = "production"
    Critical    = "true"
  }
}

# Network Module
module "network" {
  source = "../../modules/network"
  
  vpc_name = "${local.project_name}-${local.environment}"
  description = "Production VPC for Credit Scoring ML project"
  labels = local.common_tags
  
  subnets = [
    {
      name           = "${local.project_name}-${local.environment}-subnet-a"
      zone           = local.zones[0]
      v4_cidr_blocks = ["10.100.0.0/24"]
      labels = merge(local.common_tags, {
        zone = "a"
        tier = "private"
      })
    },
    {
      name           = "${local.project_name}-${local.environment}-subnet-b"
      zone           = local.zones[1]
      v4_cidr_blocks = ["10.100.1.0/24"]
      labels = merge(local.common_tags, {
        zone = "b"
        tier = "private"
      })
    },
    {
      name           = "${local.project_name}-${local.environment}-subnet-c"
      zone           = local.zones[2]
      v4_cidr_blocks = ["10.100.2.0/24"]
      labels = merge(local.common_tags, {
        zone = "c"
        tier = "private"
      })
    }
  ]
  
  # Более строгие security groups для production
  allowed_k8s_api_cidr_blocks = ["10.100.0.0/16", "${var.office_ip}/32"]  # Только внутренние IP и офис
  allowed_ssh_cidr_blocks     = ["${var.bastion_ip}/32", "${var.office_ip}/32"]
  allowed_http_cidr_blocks    = ["0.0.0.0/0"]  # Для ingress
  allowed_database_cidr_blocks = ["10.100.0.0/16", "10.20.0.0/16", "10.21.0.0/16"]

  additional_ingress_rules = [
    {
      protocol       = "TCP"
      description    = "MLflow"
      v4_cidr_blocks = ["0.0.0.0/0"]
      port           = 5000
    },
    {
      protocol       = "TCP"
      description    = "Prometheus"
      v4_cidr_blocks = ["0.0.0.0/0"]
      port           = 9090
    },
    {
      protocol       = "TCP"
      description    = "Grafana"
      v4_cidr_blocks = ["0.0.0.0/0"]
      port           = 3000
    }
  ]
}

# IAM Module
module "iam" {
  source = "../../modules/iam"
  
  project_name    = local.project_name
  environment     = local.environment
  folder_id       = var.folder_id
  labels          = local.common_tags
  
  service_accounts = {
    k8s_cluster = {
      name        = "k8s-cluster-sa"
      description = "Service account for Kubernetes cluster"
      roles = [
        "editor",
        "container-registry.images.puller",
        "kms.keys.encrypterDecrypter"
      ]
    }
    k8s_node = {
      name        = "k8s-node-sa"
      description = "Service account for Kubernetes nodes"
      roles = [
        "container-registry.images.puller",
        "monitoring.editor",
        "logging.writer",
        "kms.keys.encrypterDecrypter"
      ]
    }
    s3_access = {
      name        = "s3-access-sa"
      description = "Service account for S3 access"
      roles = [
        "storage.editor",
        "storage.uploader"
      ]
    }
  }
}

# KMS Module
module "kms" {
  source = "../../modules/storage/kms"
  
  project_name    = local.project_name
  environment     = local.environment
  folder_id       = var.folder_id
  labels          = local.common_tags
  
  keys = {
    k8s_secrets = {
      name              = "k8s-secrets-key"
      description       = "KMS key for Kubernetes secrets encryption"
      rotation_period   = "8760h"  # 1 год
    }
    database_encryption = {
      name              = "database-encryption-key"
      description       = "KMS key for database encryption"
      rotation_period   = "8760h"
    }
  }
}

# Kubernetes Module
module "kubernetes" {
  source = "../../modules/kubernetes"
  
  project_name    = local.project_name
  environment     = local.environment
  folder_id       = var.folder_id
  labels          = local.common_tags
  
  # Network
  network_id          = module.network.network_id
  subnet_id           = module.network.subnet_ids["0"]
  security_group_ids  = [module.network.security_group_ids["k8s_cluster"]]
  
  # Cluster configuration
  kubernetes_version = "1.27"
  release_channel    = "REGULAR"  # Для production использовать STABLE
  
  # Node groups
  node_groups = {
    cpu = {
      name        = "cpu-pool"
      description = "CPU optimized node pool for general workloads"
      instance_type = "standard-v2"
      platform_id   = "standard-v3"
      cpu           = 4
      memory        = 8
      disk_size     = 64
      disk_type     = "network-ssd"
      min_size      = 2
      max_size      = 5
      initial_size  = 2
      preemptible   = false
      labels = {
        "node-pool"   = "cpu"
        "environment" = local.environment
      }
    }
    gpu = {
      name        = "gpu-pool"
      description = "GPU optimized node pool for ML training"
      instance_type = "gpu-standard-v2"
      platform_id   = "gpu-standard-v3"
      cpu           = 8
      memory        = 32
      gpus          = 1
      gpu_type      = "gpu-v100"
      disk_size     = 128
      disk_type     = "network-ssd"
      min_size      = 0
      max_size      = 2
      initial_size  = 0
      preemptible   = false
      labels = {
        "node-pool"   = "gpu"
        "gpu-type"    = "v100"
        "environment" = local.environment
      }
      taints = [{
        key    = "sku"
        value  = "gpu"
        effect = "NO_SCHEDULE"
      }]
    }
  }
  
  # Service accounts
  service_account_id      = module.iam.service_account_ids["k8s_cluster"]
  node_service_account_id = module.iam.service_account_ids["k8s_node"]
  
  # KMS
  kms_key_id = module.kms.key_ids["k8s_secrets"]
  
  # Logging and monitoring
  cluster_logging_enabled    = true
  cluster_monitoring_enabled = true
}

# Database Module
module "database" {
  source = "../../modules/database"
  
  project_name    = local.project_name
  environment     = local.environment
  folder_id       = var.folder_id
  labels          = local.common_tags
  
  # Network
  network_id         = module.network.network_id
  subnet_id          = module.network.subnet_ids["0"]
  security_group_ids = [module.network.security_group_ids["database"]]
  
  # PostgreSQL
  postgresql_clusters = {
    main = {
      name        = "credit-scoring-db"
      description = "Main PostgreSQL cluster for credit scoring application"
      version     = "15"
      resources = {
        resource_preset_id = "s2.micro"
        disk_type_id       = "network-ssd"
        disk_size          = 20
      }
      databases = {
        credit_scoring = {
          name  = "credit_scoring"
          owner = "mlops"
        }
        mlflow = {
          name  = "mlflow"
          owner = "mlops"
        }
      }
      users = {
        mlops = {
          name     = "mlops"
          password = random_password.db_password.result
          grants = ["ALL"]
        }
        app = {
          name     = "app_user"
          password = random_password.app_db_password.result
          grants   = ["SELECT", "INSERT", "UPDATE", "DELETE"]
        }
      }
      # Backup configuration
      backup_window_start = {
        hours   = 23
        minutes = 0
      }
      maintenance_window = {
        type = "WEEKLY"
        day  = "SAT"
        hour = 2
      }
    }
  }
  
  # Redis (optional)
  redis_clusters = {
    cache = {
      name        = "credit-scoring-cache"
      description = "Redis cluster for caching"
      version     = "7.0"
      resources = {
        resource_preset_id = "hm2.nano"
        disk_size          = 8
      }
      persistence_mode = "ON"
    }
  }
}

# Storage Module
module "storage" {
  source = "../../modules/storage"
  
  project_name    = local.project_name
  environment     = local.environment
  folder_id       = var.folder_id
  labels          = local.common_tags
  
  s3_buckets = {
    terraform-state = {
      name        = "mlops-terraform-state"
      description = "Bucket for Terraform state files"
      versioning_enabled = true
      lifecycle_rules = [
        {
          id      = "state-retention"
          enabled = true
          expiration = {
            days = 365
          }
          noncurrent_version_expiration = {
            days = 30
          }
        }
      ]
    }
    ml-models = {
      name        = "credit-scoring-models"
      description = "Bucket for ML models and artifacts"
      versioning_enabled = true
      lifecycle_rules = [
        {
          id      = "model-retention"
          enabled = true
          expiration = {
            days = 180
          }
        }
      ]
    }
    data-storage = {
      name        = "credit-scoring-data"
      description = "Bucket for training and reference data"
      versioning_enabled = true
    }
  }
}

# Container Registry
resource "yandex_container_registry" "main" {
  name      = "${local.project_name}-registry"
  folder_id = var.folder_id
  labels    = local.common_tags
}

# Outputs
output "cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = module.kubernetes.cluster_external_endpoint
  sensitive   = true
}

output "cluster_id" {
  description = "Kubernetes cluster ID"
  value       = module.kubernetes.cluster_id
}

output "registry_id" {
  description = "Container registry ID"
  value       = yandex_container_registry.main.id
}

output "database_host" {
  description = "PostgreSQL database host"
  value       = module.database.postgresql_hosts["main"]
  sensitive   = true
}

output "redis_host" {
  description = "Redis host"
  value       = try(module.database.redis_hosts["cache"], null)
  sensitive   = true
}

output "s3_buckets" {
  description = "S3 bucket names"
  value       = module.storage.bucket_names
}

output "kubeconfig" {
  description = "Kubeconfig for cluster access"
  value       = module.kubernetes.kubeconfig
  sensitive   = true
}

output "database_passwords" {
  description = "Database passwords"
  value = {
    mlops     = random_password.db_password.result
    app_user  = random_password.app_db_password.result
  }
  sensitive = true
}

# Random passwords
resource "random_password" "db_password" {
  length  = 16
  special = true
  override_special = "_%@"
}

resource "random_password" "app_db_password" {
  length  = 16
  special = true
  override_special = "_%@"
}