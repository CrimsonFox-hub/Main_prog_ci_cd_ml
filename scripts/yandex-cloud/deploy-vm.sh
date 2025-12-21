#!/bin/bash
# Деплой проекта на Yandex Cloud VM

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Деплой MLOps проекта в Yandex Cloud${NC}"

# Проверка наличия yc CLI
if ! command -v yc &> /dev/null; then
    echo -e "${RED}Ошибка: yc CLI не установлен${NC}"
    echo "Установите: https://cloud.yandex.ru/docs/cli/quickstart"
    exit 1
fi

# Проверка наличия SSH ключа
SSH_KEY_PATH="$HOME/.ssh/id_ed25519_yc"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo -e "${YELLOW}Создание SSH ключа для Yandex Cloud...${NC}"
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "mlops-yc-$(date +%Y%m%d)"
fi

# Параметры
VM_NAME="mlops-credit-scoring"
ZONE="ru-central1-a"
IMAGE_ID="fd8vmcue6laajqo4oeq0"  # Ubuntu 22.04 LTS
CORES=4
MEMORY=8GB
DISK_SIZE=30GB

echo -e "${GREEN}Создание виртуальной машины...${NC}"

# Создание VM
yc compute instance create \
    --name "$VM_NAME" \
    --zone "$ZONE" \
    --platform-id "standard-v3" \
    --cores "$CORES" \
    --memory "$MEMORY" \
    --network-interface "subnet-name=default-ru-central1-a,nat-ip-version=ipv4" \
    --create-boot-disk "size=$DISK_SIZE,image-id=$IMAGE_ID" \
    --metadata-from-file "ssh-keys=$SSH_KEY_PATH.pub" \
    --preemptible

# Получение публичного IP
VM_IP=$(yc compute instance get "$VM_NAME" --format json | jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address')

echo -e "${GREEN}VM создана: $VM_NAME (IP: $VM_IP)${NC}"

# Ожидание доступности SSH
echo -e "${YELLOW}Ожидание доступности SSH...${NC}"
sleep 30

# Проверка доступности
until nc -z "$VM_IP" 22; do
    echo "Ожидание SSH..."
    sleep 10
done

# Настройка VM
echo -e "${GREEN}Настройка виртуальной машины...${NC}"
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no yc-user@"$VM_IP" << 'EOF'
    # Обновление системы
    sudo apt-get update
    sudo apt-get upgrade -y
    
    # Установка Docker
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Установка Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Настройка пользователя
    sudo usermod -aG docker $USER
    sudo systemctl enable docker
    sudo systemctl start docker
    
    echo "Docker и Docker Compose установлены"
EOF

# Копирование проекта
echo -e "${GREEN}Копирование проекта на VM...${NC}"
scp -i "$SSH_KEY_PATH" -r ./* yc-user@"$VM_IP":/home/yc-user/mlops-project/

# Запуск проекта
echo -e "${GREEN}Запуск проекта...${NC}"
ssh -i "$SSH_KEY_PATH" yc-user@"$VM_IP" << EOF
    cd /home/yc-user/mlops-project
    
    # Создание необходимых директорий
    mkdir -p logs configs models/trained data/processed
    
    # Запуск docker-compose
    docker-compose -f docker-compose.yc.yml build
    docker-compose -f docker-compose.yc.yml up -d
    
    echo "Проект запущен!"
    
    # Показ информации о сервисах
    echo -e "\nДоступные сервисы:"
    echo "API: http://$VM_IP:8000"
    echo "MLflow: http://$VM_IP:5000"
    echo "Grafana: http://$VM_IP:3000"
    echo "MinIO Console: http://$VM_IP:9001"
EOF

echo -e "${GREEN}Деплой завершен!${NC}"
echo -e "\n${YELLOW}Ссылки на сервисы:${NC}"
echo "API: http://$VM_IP:8000"
echo "MLflow: http://$VM_IP:5000"
echo "Grafana: http://$VM_IP:3000 (admin/admin)"
echo "MinIO Console: http://$VM_IP:9001 (minioadmin/minioadmin)"

# Проверка доступности
echo -e "\n${YELLOW}Проверка доступности сервисов...${NC}"
sleep 10

check_service() {
    local service=$1
    local port=$2
    local url=$3
    
    if curl -s -f "http://$VM_IP:$port" > /dev/null; then
        echo -e "  ✓ $service доступен: $url"
    else
        echo -e "  ✗ $service недоступен"
    fi
}

check_service "API" "8000" "http://$VM_IP:8000"
check_service "MLflow" "5000" "http://$VM_IP:5000"
check_service "Grafana" "3000" "http://$VM_IP:3000"
check_service "MinIO" "9001" "http://$VM_IP:9001"