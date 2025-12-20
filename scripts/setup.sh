#!/bin/bash
set -e

echo "🚀 Начало развертывания MLOps проекта..."

# 1. Проверка зависимостей
check_dependencies() {
    echo "🔍 Проверка зависимостей..."
    command -v docker >/dev/null 2>&1 || { echo "❌ Docker не установлен"; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl не установлен"; exit 1; }
    command -v terraform >/dev/null 2>&1 || { echo "❌ Terraform не установлен"; exit 1; }
    command -v yc >/dev/null 2>&1 || { echo "❌ YC CLI не установлен"; exit 1; }
    echo "✅ Все зависимости установлены"
}

# 2. Настройка облака
setup_cloud() {
    echo "☁️  Настройка облачной инфраструктуры..."
    cd infrastructure/environments/staging
    terraform init
    terraform apply -auto-approve
    cd ../../..
}

# 3. Сборка и загрузка образов
build_and_push() {
    echo "🐳 Сборка Docker образов..."
    docker-compose build
    docker push cr.yandexcloud.net/$REGISTRY_ID/credit-scoring-api:latest
    docker push cr.yandexcloud.net/$REGISTRY_ID/credit-scoring-training:latest
}

# 4. Развертывание в Kubernetes
deploy_k8s() {
    echo "⚙️  Развертывание в Kubernetes..."
    kubectl apply -f kubernetes/ -R
    echo "⏳ Ожидание запуска pods..."
    kubectl wait --for=condition=ready pod -l app=credit-scoring-api --timeout=300s
}

# Основной скрипт
main() {
    check_dependencies
    setup_cloud
    build_and_push
    deploy_k8s
    echo "✅ Развертывание завершено!"
    echo "📊 Доступные сервисы:"
    echo "   - API: http://api.example.com"
    echo "   - Grafana: http://grafana.example.com"
    echo "   - Airflow: http://airflow.example.com"
}

main "$@"