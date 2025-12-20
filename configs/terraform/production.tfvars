# Terraform переменные для production
environment = "production"

# Yandex Cloud
yc_cloud_id = "your-cloud-id"
yc_folder_id = "your-folder-id"
yc_token = "your-token"

# Network
vpc_name = "credit-scoring-vpc"
vpc_subnets = [
  {
    zone           = "ru-central1-a"
    v4_cidr_blocks = ["10.0.1.0/24"]
  }
]

# Kubernetes
k8s_cluster_name = "credit-scoring-cluster"
k8s_version = "1.27"

node_groups = {
  cpu = {
    instance_type = "standard-v2"
    cores         = 4
    memory        = 8
    disk_size     = 50
    min_size      = 2
    max_size      = 5
  }
  gpu = {
    instance_type = "gpu-standard-v2-t4"
    cores         = 8
    memory        = 32
    disk_size     = 100
    gpus          = 1
    min_size      = 0
    max_size      = 2
  }
}

# Database
postgres_version = "15"
postgres_disk_size = 50
postgres_user = "mlops"
postgres_db = "credit_scoring"

# Redis
redis_version = "7"
redis_memory = 4

# Object Storage
s3_bucket_name = "credit-scoring-models"

# Monitoring
enable_monitoring = true
prometheus_disk_size = 50
grafana_admin_password = "change-me-in-production"