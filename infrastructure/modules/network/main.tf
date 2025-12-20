# Модуль для создания сети в Yandex Cloud
terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.95.0"
    }
  }
}

resource "yandex_vpc_network" "this" {
  name        = var.vpc_name
  description = var.description
  labels      = var.labels
}

resource "yandex_vpc_subnet" "this" {
  for_each = { for idx, subnet in var.subnets : idx => subnet }

  name           = each.value.name
  description    = each.value.description
  zone           = each.value.zone
  v4_cidr_blocks = each.value.v4_cidr_blocks
  network_id     = yandex_vpc_network.this.id
  route_table_id = try(each.value.route_table_id, null)

  dynamic "dhcp_options" {
    for_each = each.value.domain_name != null ? [1] : []
    content {
      domain_name         = each.value.domain_name
      domain_name_servers = each.value.domain_name_servers
    }
  }

  labels = merge(var.labels, each.value.labels)
}

# NAT Gateway для выхода в интернет
resource "yandex_vpc_gateway" "nat_gateway" {
  name = "${var.vpc_name}-nat-gateway"
  shared_egress_gateway {}
}

resource "yandex_vpc_route_table" "nat_route" {
  network_id = yandex_vpc_network.this.id
  name       = "${var.vpc_name}-nat-route"

  static_route {
    destination_prefix = "0.0.0.0/0"
    gateway_id         = yandex_vpc_gateway.nat_gateway.id
  }

  labels = var.labels
}

# Security Groups
resource "yandex_vpc_security_group" "k8s_cluster" {
  name        = "${var.vpc_name}-k8s-cluster-sg"
  description = "Security group for Kubernetes cluster"
  network_id  = yandex_vpc_network.this.id
  labels      = var.labels

  # Allow all internal traffic
  ingress {
    protocol          = "ANY"
    description       = "Intra-cluster communication"
    v4_cidr_blocks    = var.allowed_internal_cidr_blocks
  }

  # Allow Kubernetes API
  ingress {
    protocol       = "TCP"
    description    = "Kubernetes API server"
    port           = 6443
    v4_cidr_blocks = var.allowed_k8s_api_cidr_blocks
  }

  # Allow SSH
  ingress {
    protocol       = "TCP"
    description    = "SSH"
    port           = 22
    v4_cidr_blocks = var.allowed_ssh_cidr_blocks
  }

  # Allow HTTP/HTTPS
  ingress {
    protocol       = "TCP"
    description    = "HTTP"
    port           = 80
    v4_cidr_blocks = var.allowed_http_cidr_blocks
  }

  ingress {
    protocol       = "TCP"
    description    = "HTTPS"
    port           = 443
    v4_cidr_blocks = var.allowed_http_cidr_blocks
  }

  # Allow NodePort range
  ingress {
    protocol       = "TCP"
    description    = "NodePort services"
    from_port      = 30000
    to_port        = 32767
    v4_cidr_blocks = var.allowed_nodeport_cidr_blocks
  }

  # Egress - allow all outgoing
  egress {
    protocol       = "ANY"
    description    = "Outgoing traffic"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  dynamic "ingress" {
    for_each = var.additional_ingress_rules
    content {
      protocol       = ingress.value.protocol
      description    = ingress.value.description
      v4_cidr_blocks = ingress.value.v4_cidr_blocks
      port           = try(ingress.value.port, null)
      from_port      = try(ingress.value.from_port, null)
      to_port        = try(ingress.value.to_port, null)
    }
  }

  dynamic "egress" {
    for_each = var.additional_egress_rules
    content {
      protocol       = egress.value.protocol
      description    = egress.value.description
      v4_cidr_blocks = egress.value.v4_cidr_blocks
      port           = try(egress.value.port, null)
      from_port      = try(egress.value.from_port, null)
      to_port        = try(egress.value.to_port, null)
    }
  }
}

resource "yandex_vpc_security_group" "database" {
  name        = "${var.vpc_name}-database-sg"
  description = "Security group for databases"
  network_id  = yandex_vpc_network.this.id
  labels      = var.labels

  # Allow PostgreSQL
  ingress {
    protocol       = "TCP"
    description    = "PostgreSQL"
    port           = 5432
    v4_cidr_blocks = var.allowed_database_cidr_blocks
  }

  # Allow Redis
  ingress {
    protocol       = "TCP"
    description    = "Redis"
    port           = 6379
    v4_cidr_blocks = var.allowed_database_cidr_blocks
  }

  egress {
    protocol       = "ANY"
    description    = "Outgoing traffic"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}