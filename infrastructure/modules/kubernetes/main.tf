# infrastructure/modules/kubernetes/main.tf
resource "yandex_kubernetes_cluster" "credit_scoring" {
name = "credit-scoring-${var.environment}"
network_id = var.network_id

master {
    version "1.24"
    public_ip = true

master_location {
    zone = var.zone
    subnet_id = var.subnet_id
                }
}
service_account_id = yandex_iam_service_account.cluster.id
node_service_account_id = yandex_iam_service_account.nodes.id

kms_provider {
    key_id = yandex_kms_symmetric_key.kms_key.id
            }
}

resource "yandex_kubernetes_node_group" "cpu_nodes" {

    cluster_id = yandex_kubernetes_cluster.credit_scoring.id
    name = "cpu-nodes-${var.environment}"

    instance_template {
        platform_id = "standard-v2"

    resources {
        memory = 8
        cores = 4
    }

    boot_disk {
        type = "network-ssd"
        size = 64
    }

    scheduling_policy {
        preemptible = var.environment != "production"
    }
}

    scale_policy {
        auto_scale {
            min = 2
            max =10
            initial = 2
        }
    }
}