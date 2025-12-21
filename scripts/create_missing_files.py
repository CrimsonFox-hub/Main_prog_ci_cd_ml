import os
from pathlib import Path

def create_missing_files():
    """Создание недостающих файлов в проекте"""
    project_root = Path(__file__).parent.parent
    
    # Создаем структуру директорий
    dirs_to_create = [
        "src/ml_pipeline/inference",
        "scripts/windows",
        "scripts/linux",
        "configs/grafana/datasources",
        "configs/grafana/dashboards"
    ]
    
    for dir_path in dirs_to_create:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Создаем недостающие файлы
    files = {
        "src/ml_pipeline/inference/explainer.py": '''"""
Объяснение предсказаний модели.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PredictionExplainer:
    """Заглушка для объяснителя предсказаний"""
    
    def __init__(self, model=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names or []
    
    def explain_prediction(self, features, prediction, method="simple"):
        """Упрощенное объяснение предсказания"""
        return {
            "prediction": float(prediction),
            "feature_importance": {f"feature_{i}": float(abs(val)) 
                                  for i, val in enumerate(features)},
            "method": method
        }

def get_explainer():
    """Получение глобального экземпляра объяснителя"""
    return PredictionExplainer()

def init_explainer(model, feature_names, X_train=None):
    """Инициализация объяснителя"""
    return PredictionExplainer(model, feature_names)
''',
        
        "scripts/windows/health-check.ps1": '''# Проверка здоровья сервисов для Windows
Write-Host "Проверка здоровья сервисов MLOps Credit Scoring" -ForegroundColor Green

$services = @(
    @{Name="API"; Port=8000; Path="/health"},
    @{Name="MLflow"; Port=5000; Path=""},
    @{Name="MinIO Console"; Port=9001; Path=""},
    @{Name="Grafana"; Port=3000; Path=""},
    @{Name="Prometheus"; Port=9090; Path=""},
    @{Name="Loki"; Port=3100; Path="/ready"}
)

foreach ($service in $services) {
    $url = "http://localhost:$($service.Port)$($service.Path)"
    try {
        $response = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 5 -ErrorAction Stop
        Write-Host "✓ $($service.Name) доступен ($url)" -ForegroundColor Green
    } catch {
        Write-Host "✗ $($service.Name) недоступен ($url)" -ForegroundColor Red
    }
}

# Проверка Docker контейнеров
Write-Host "`nПроверка Docker контейнеров..." -ForegroundColor Yellow
docker-compose ps
''',
        
        "scripts/linux/health-check.sh": '''#!/bin/bash
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

echo -e "\nПроверка Docker контейнеров..."
docker-compose ps
'''
    }
    
    created = 0
    for file_path, content in files.items():
        full_path = project_root / file_path
        if not full_path.exists():
            full_path.write_text(content, encoding='utf-8')
            print(f"✓ Создан: {file_path}")
            created += 1
    
    print(f"\nСоздано {created} файлов")

if __name__ == "__main__":
    create_missing_files()