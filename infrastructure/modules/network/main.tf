module "vpc" {
  source = "terraform-yandex-modules/vpc/yandex"
  
  name = "${var.project_name}-vpc"
  folder_id = var.folder_id
  description = "VPC for ML project"
  
  subnets = [
    {
      zone           = var.zone
      v4_cidr_blocks = ["10.0.1.0/24"]
      route_table_id = yandex_vpc_route_table.nat.id
    }
  ]
}

resource "yandex_vpc_security_group" "k8s" {
  name        = "${var.project_name}-k8s-sg"
  description = "Security group for Kubernetes cluster"
  network_id  = module.vpc.network_id
  
  ingress {
    protocol       = "ANY"
    description    = "Intra-cluster communication"
    v4_cidr_blocks = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
  }
  
  ingress {
    protocol       = "TCP"
    port           = 80
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    protocol       = "TCP"
    port           = 443
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}