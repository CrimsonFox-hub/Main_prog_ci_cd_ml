output "network_id" {
  description = "ID of the created VPC network"
  value       = yandex_vpc_network.this.id
}

output "subnet_ids" {
  description = "Map of subnet IDs"
  value       = { for k, v in yandex_vpc_subnet.this : k => v.id }
}

output "security_group_ids" {
  description = "Map of security group IDs"
  value = {
    k8s_cluster = yandex_vpc_security_group.k8s_cluster.id
    database    = yandex_vpc_security_group.database.id
  }
}

output "nat_gateway_id" {
  description = "ID of the NAT gateway"
  value       = yandex_vpc_gateway.nat_gateway.id
}

output "route_table_id" {
  description = "ID of the route table"
  value       = yandex_vpc_route_table.nat_route.id
}