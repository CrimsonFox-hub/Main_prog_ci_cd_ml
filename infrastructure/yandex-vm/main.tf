# infrastructure/yandex-vm/main.tf
terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.47.0"
    }
  }
}

provider "yandex" {
  zone = "ru-central1-a"
}

resource "yandex_vpc_network" "mlops-network" {
  name = "mlops-network"
}

resource "yandex_vpc_subnet" "mlops-subnet" {
  name           = "mlops-subnet"
  zone           = "ru-central1-a"
  network_id     = yandex_vpc_network.mlops-network.id
  v4_cidr_blocks = ["192.168.10.0/24"]
}

resource "yandex_compute_instance" "mlops-vm" {
  name        = "mlops-vm"
  platform_id = "standard-v3"
  zone        = "ru-central1-a"

  resources {
    cores  = 4
    memory = 8
  }

  boot_disk {
    initialize_params {
      image_id = "fd8vmcue6la********" # Ubuntu 22.04 LTS
      size     = 30
    }
  }

  network_interface {
    subnet_id = yandex_vpc_subnet.mlops-subnet.id
    nat       = true # Публичный IP
  }

  metadata = {
    user-data = "#cloud-config\nusers:\n  - name: ubuntu\n    groups: sudo\n    shell: /bin/bash\n    sudo: 'ALL=(ALL) NOPASSWD:ALL'\n    ssh_authorized_keys:\n      - ${file("~/.ssh/id_rsa.pub")}"
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/id_rsa")
    host        = self.network_interface[0].nat_ip_address
  }

  provisioner "remote-exec" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install -y docker.io docker-compose",
      "sudo usermod -aG docker ubuntu"
    ]
  }
}

output "vm_public_ip" {
  value = yandex_compute_instance.mlops-vm.network_interface[0].nat_ip_address
}