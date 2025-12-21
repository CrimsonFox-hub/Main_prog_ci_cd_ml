#!/bin/bash
# Проверка здоровья сервисов для Linux

echo "Проверка здоровья сервисов MLOps Credit Scoring"

services=(
    "API:8000:/health"
    "MLflow:5000:"
    "MinIO:9001:"
    "Grafana:3000:"
    "Prometheus:9090:"
    "Loki:3100:/ready"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port path <<< "$service"
    url="http://localhost:${port}${path}"
    
    if curl -s -f "$url" > /dev/null; then
        echo "✓ $name доступен ($url)"
    else
        echo "✗ $name недоступен ($url)"
    fi
done

echo -e "
Проверка Docker контейнеров..."
docker-compose ps
