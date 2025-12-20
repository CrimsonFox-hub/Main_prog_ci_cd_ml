terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.92.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.23.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.11.0"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14.0"
    }
  }
}

provider "yandex" {
  token     = var.yc_token
  cloud_id  = var.cloud_id
  folder_id = var.folder_id
  zone      = var.zone
}

provider "kubernetes" {
  host                   = yandex_kubernetes_cluster.mlops_cluster.master.0.external_v4_endpoint
  cluster_ca_certificate = base64decode(yandex_kubernetes_cluster.mlops_cluster.master.0.cluster_ca_certificate)
  token                  = var.k8s_token
}

provider "helm" {
  kubernetes {
    host                   = yandex_kubernetes_cluster.mlops_cluster.master.0.external_v4_endpoint
    cluster_ca_certificate = base64decode(yandex_kubernetes_cluster.mlops_cluster.master.0.cluster_ca_certificate)
    token                  = var.k8s_token
  }
}