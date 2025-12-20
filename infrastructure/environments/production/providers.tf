# Providers configuration for staging environment

provider "yandex" {
  token     = var.yc_token
  cloud_id  = var.cloud_id
  folder_id = var.folder_id
  zone      = var.zone
  
  endpoint = "api.cloud.yandex.net:443"
}

provider "kubernetes" {
  host  = module.kubernetes.cluster_external_endpoint
  token = data.yandex_client_config.client.iam_token
  
  cluster_ca_certificate = base64decode(
    module.kubernetes.cluster_ca_certificate
  )
}

provider "helm" {
  kubernetes {
    host  = module.kubernetes.cluster_external_endpoint
    token = data.yandex_client_config.client.iam_token
    
    cluster_ca_certificate = base64decode(
      module.kubernetes.cluster_ca_certificate
    )
  }
}

provider "kubectl" {
  host  = module.kubernetes.cluster_external_endpoint
  token = data.yandex_client_config.client.iam_token
  
  cluster_ca_certificate = base64decode(
    module.kubernetes.cluster_ca_certificate
  )
  
  load_config_file = false
}

data "yandex_client_config" "client" {}