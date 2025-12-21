#!/usr/bin/env python3
"""
Скрипт для запуска всей системы Credit Scoring
"""
import subprocess
import time
import sys
import os
from pathlib import Path
import webbrowser

try:
    import yaml
except ImportError:
    print("Установка модуля PyYAML...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

try:
    import requests
except ImportError:
    print("Установка модуля requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

def print_banner():
    """Печать баннера системы"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║        CREDIT SCORING MLOps SYSTEM v2.0                  ║
    ║        Нейронная сеть для кредитного скоринга           ║
    ║        Промышленное ML с полным стеком MLOps            ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_command(command, description, wait=True):
    """Запуск команды"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")
    
    try:
        if wait:
            result = subprocess.run(command, shell=True, check=True)
            return result.returncode == 0
        else:
            # Для Windows
            if sys.platform == "win32":
                subprocess.Popen(f'start cmd /k "{command}"', shell=True)
            # Для Linux/Mac
            else:
                subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{command}; exec bash'])
            return True
    except Exception as e:
        print(f" Ошибка: {e}")
        return False

def check_service(url, service_name):
    """Проверка доступности сервиса"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"    {service_name}: {url}")
            return True
        else:
            print(f"     {service_name}: недоступен (код {response.status_code})")
            return False
    except:
        print(f"    {service_name}: недоступен")
        return False
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"    {service_name}: {url}")
            return True
        else:
            print(f"     {service_name}: недоступен (код {response.status_code})")
            return False
    except:
        print(f"    {service_name}: недоступен")
        return False

def setup_project():
    """Настройка проекта"""
    print("\n1.   Настройка проекта...")
    
    # Создание директорий
    directories = [
        'models', 'data/raw', 'data/processed', 'reports',
        'configs', 'logs', 'mlruns', 'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Создание конфигурационных файлов
    configs = {
        'configs/training_config.yaml': """# Конфигурация обучения нейронной сети
data_path: "data/processed/train.csv"
target_column: "default"

# Признаки
numerical_features:
  - "duration"
  - "credit_amount"
  - "age"
  - "installment_commitment"
  - "residence_since"
  - "existing_credits"
  - "num_dependents"

categorical_features:
  - "checking_status"
  - "credit_history"
  - "purpose"
  - "savings_status"
  - "employment"

# Конфигурация модели
model_config:
  type: "simple"
  hidden_layers: [128, 64, 32]
  dropout_rate: 0.3

# Параметры обучения
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10

# MLflow
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "credit_scoring_neural_network"
""",
        'configs/api_config.yaml': """# Конфигурация API
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

model:
  path: "models/credit_scoring.onnx"
  threshold: 0.5

services:
  mlflow: "http://localhost:5000"
  grafana: "http://localhost:3000"
  minio: "http://localhost:9001"
"""
    }
    
    for config_file, content in configs.items():
        if not Path(config_file).exists():
            with open(config_file, 'w') as f:
                f.write(content)
            print(f"    Создан: {config_file}")
    
    return True

def install_dependencies():
    """Установка зависимостей"""
    print("\n2.  Установка зависимостей...")
    
    if run_command("pip install -r requirements.txt", "Установка Python зависимостей"):
        print("   ✅ Зависимости установлены")
        return True
    else:
        print("    Ошибка установки зависимостей")
        return False
    
    

def download_sample_data():
    """Загрузка тестовых данных"""
    print("\n3.  Загрузка тестовых данных...")
    
    # Создание синтетических данных для демонстрации
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'duration': np.random.randint(1, 72, n_samples),
        'credit_amount': np.random.randint(500, 20000, n_samples),
        'age': np.random.randint(18, 75, n_samples),
        'installment_commitment': np.random.randint(1, 4, n_samples),
        'residence_since': np.random.randint(1, 10, n_samples),
        'existing_credits': np.random.randint(1, 4, n_samples),
        'num_dependents': np.random.randint(0, 3, n_samples),
        'checking_status': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
        'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
        'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'], n_samples),
        'savings_status': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
        'employment': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Сохранение данных
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/processed/train.csv', index=False)
    
    print(f"   ✅ Созданы тестовые данные: {len(df)} записей")
    print(f"   ✅ Распределение целевой переменной:")
    print(f"      - Хорошие заемщики: {(df['default'] == 0).sum()}")
    print(f"      - Плохие заемщики: {(df['default'] == 1).sum()}")
    
    return True

def train_model():
    """Обучение модели"""
    print("\n4.  Обучение нейронной сети...")
    
    if run_command("python main_pipeline.py", "Обучение модели", wait=True):
        # Проверка наличия моделей
        models_exist = all([
            Path('models/credit_scoring_nn.pth').exists(),
            Path('models/credit_scoring.onnx').exists()
        ])
        
        if models_exist:
            print("   ✅ Модель обучена и сохранена")
            return True
        else:
            print("     Модели не найдены после обучения")
            return False
    else:
        print("    Ошибка обучения модели")
        return False

def start_services():
    """Запуск сервисов"""
    print("\n5.  Запуск сервисов...")
    
    # Проверка наличия Docker
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.DEVNULL)
    except:
        print("   Docker не установлен. Пропуск запуска Docker сервисов.")
        return False
    
    # Запуск Docker Compose
    if run_command("docker-compose up -d", "Запуск Docker сервисов", wait=True):
        print("    Ожидание запуска сервисов (30 секунд)...")
        time.sleep(30)
        
        # Проверка сервисов
        print("\n   🔍 Проверка доступности сервисов...")
        services = {
            "MLflow": "http://localhost:5000",
            "MinIO Console": "http://localhost:9001",
        }
        
        all_services_up = True
        for name, url in services.items():
            if not check_service(url, name):
                all_services_up = False
        
        if all_services_up:
            print("    Все сервисы запущены")
            return True
        else:
            print("     Некоторые сервисы не доступны")
            return True  # Все равно продолжаем
    else:
        print("    Ошибка запуска Docker сервисов")
        return False

def start_api():
    """Запуск API"""
    print("\n6. ⚡ Запуск FastAPI сервера...")
    
    if run_command("python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload", "FastAPI сервер", wait=False):
        print("   ⏳ Ожидание запуска API (10 секунд)...")
        time.sleep(10)
        
        if check_service("http://localhost:8000/health", "API сервер"):
            print("    API сервер запущен")
            return True
        else:
            print("     API сервер не доступен")
            return False
    else:
        print("    Ошибка запуска API сервера")
        return False

def open_browser():
    """Открытие браузера"""
    print("\n7.  Открытие веб-интерфейса...")
    
    urls = [
        "http://localhost:8000",
        "http://localhost:8000/api/docs",
        "http://localhost:5000",
    ]
    
    for url in urls:
        try:
            webbrowser.open(url)
            print(f"    Открыто: {url}")
            time.sleep(1)
        except:
            print(f"     Не удалось открыть: {url}")
    
    return True

def print_summary():
    """Печать итоговой информации"""
    print(f" Главная страница:      http://localhost:8000")
    print(f" Демо модель:           http://localhost:8000/demo")
    print(f" API документация:      http://localhost:8000/api/docs")
    print(f" MLflow:                http://localhost:5000")
    print(f" MinIO Console:         http://localhost:9001")
    

    print("MinIO: minioadmin / minioadmin")
    print("Переобучить модель:     python main_pipeline.py")
    print("Остановить сервисы:     docker-compose down")
    print("Просмотр логов API:     docker-compose logs -f api")
    

    print("Мониторинг дрифта:      python src/ml_pipeline/monitoring/drift_detection.py")
    print("Дашборды:               python src/ml_pipeline/monitoring/evidently_dashboard.py")
    
    print("1. Нажмите Ctrl+C в этом окне")
    print("2. Выполните: docker-compose down")
    print("3. Закройте все терминалы")

def main():
    """Основная функция"""
    print_banner()
    
    steps = [
        ("Настройка проекта", setup_project),
        ("Установка зависимостей", install_dependencies),
        ("Загрузка тестовых данных", download_sample_data),
        ("Обучение модели", train_model),
        ("Запуск сервисов", start_services),
        ("Запуск API", start_api),
        ("Открытие браузера", open_browser),
    ]
    
    successful_steps = []
    
    for step_name, step_func in steps:
        try:
            if step_func():
                successful_steps.append(step_name)
            else:
                print(f"\n  Шаг '{step_name}' завершился с предупреждениями")
        except Exception as e:
            print(f"\n Ошибка в шаге '{step_name}': {e}")
            print("Продолжаем выполнение...")
    
    print_summary()
    
    # Бесконечный цикл для удержания скрипта
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Остановка системы...")
        
        # Остановка Docker сервисов
        print(" Остановка Docker сервисов...")
        subprocess.run(["docker-compose", "down"], capture_output=True)
        
        print("\n Система остановлена. До свидания!")

if __name__ == "__main__":
    main()