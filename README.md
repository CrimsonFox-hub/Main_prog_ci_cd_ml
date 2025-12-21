Credit Scoring MLOps System
Проект представляет собой полную промышленную систему для кредитного скоринга с использованием нейронных сетей и современных MLOps практик.

🎯 Особенности
Нейронная сеть на PyTorch с архитектурой 128-64-32

ONNX формат для production инференса

Quantization & Pruning для оптимизации модели

Полный MLOps стек: MLflow, Grafana, Prometheus, Loki

Docker Compose для оркестрации сервисов

FastAPI с интерактивным веб-интерфейсом

Автоматический мониторинг дрифта данных (Evidently AI)

CI/CD с GitHub Actions

Infrastructure as Code с Terraform

🏗️ Архитектура системы
```text
┌─────────────────────────────────────────────────────────────┐
│                     Пользовательские запросы                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     FastAPI (REST API)                      │
│                     Порт: 8000                              │
└─────────────┬──────────────────────┬───────────────────────┘
              │                      │
    ┌─────────▼──────────┐  ┌───────▼────────┐
    │   ML инференс      │  │  Данные/кэш    │
    │   (ONNX Runtime)   │  │  PostgreSQL    │
    └─────────┬──────────┘  │  Redis         │
              │             └───────┬────────┘
    ┌─────────▼──────────┐          │
    │   Модель           │  ┌───────▼────────┐
    │   PyTorch → ONNX   │  │  Мониторинг    │
    └─────────┬──────────┘  │  Prometheus    │
              │             │  Grafana       │
    ┌─────────▼──────────┐  │  Loki          │
    │   MLflow           │  └────────────────┘
    │   Трекинг моделей  │
    └────────────────────┘
```
🚀 Быстрый старт (Локальная установка)\
Предварительные требования\
Windows: Docker Desktop, Git, Python 3.9+

Linux/Mac: Docker Engine, Docker Compose, Git, Python 3.9+

Установка и запуск (Windows)\
```powershell
# 1. Клонируйте репозиторий
git clone https://github.com/CrimsonFox-hub/Main_prog_ci_cd_ml.git
cd Main_prog_ci_cd_ml

# 2. Создайте недостающие файлы конфигурации
python scripts/create_missing_files.py

# 3. Запустите все сервисы одной командой
make run-local

# ИЛИ для полного пайплайна
make full-pipeline
```
Установка и запуск (Linux/Mac)
```bash
# 1. Клонируйте репозиторий
git clone https://github.com/CrimsonFox-hub/Main_prog_ci_cd_ml.git
cd Main_prog_ci_cd_ml

# 2. Настройка окружения
make setup-local

# 3. Запуск всех сервисов
make up-all

# 4. Обучение и подготовка модели
make pipeline
🌐 Доступные сервисы
После запуска все сервисы будут доступны по следующим адресам:

Сервис	Назначение	URL	Порт	Учетные данные
API	Кредитный скоринг	http://localhost:8000	8000	-
MLflow	Трекинг экспериментов	http://localhost:5000	5000	-
Grafana	Дашборды и визуализация	http://localhost:3000	3000	admin/admin
MinIO Console	Объектное хранилище	http://localhost:9001	9001	minioadmin/minioadmin
Loki	Просмотр логов	http://localhost:3100	3100	-
Prometheus	Сбор метрик	http://localhost:9090	9090	-
🧠 Архитектура модели
Нейронная сеть
python
CreditScoringNN(
    input_size=20,           # Количество признаков
    hidden_layers=[128, 64, 32],
    dropout_rate=0.3,
    output_size=1           # Бинарная классификация
)
Оптимизации модели
Pruning: Удаление 20% наименее значимых весов

Quantization: 8-bit динамическое квантование

ONNX: Конвертация в оптимизированный production-формат

Метрики производительности
Точность: ~85%

Время инференса: <5 мс

Размер модели: <2 MB

Поддержка batch-обработки: до 1000 запросов/сек

📊 API документация
Основные эндпоинты
Предсказание для одного клиента
http
POST /api/v1/predict
Content-Type: application/json

{
  "age": 35,
  "income": 75000,
  "credit_score": 720,
  "loan_amount": 25000,
  "employment_years": 8,
  "debt_to_income": 0.25,
  "has_default": false,
  "loan_purpose": "home"
}
Ответ:

json
{
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "low",
  "recommendation": "Кредит рекомендуется к одобрению",
  "explanation": {
    "top_features": ["credit_score", "income", "debt_to_income"],
    "confidence": 0.92
  }
}
Пакетное предсказание
http
POST /api/v1/batch_predict
Content-Type: application/json

{
  "data": [
    { "age": 35, "income": 75000, ... },
    { "age": 42, "income": 65000, ... }
  ]
}
Проверка здоровья системы
http
GET /health
Информация о модели
http
GET /api/v1/models/info
Дополнительные эндпоинты
http
GET /api/v1/system/metrics     # Метрики Prometheus
GET /api/v1/services          # Список сервисов
POST /api/v1/feedback        # Отправка обратной связи
GET /api/v1/version          # Версия API
🐳 Docker сервисы
В проекте используются следующие Docker-сервисы:

Сервис	Порт	Описание	Образ
api	8000	FastAPI приложение	credit-scoring-api:latest
mlflow	5000	Трекинг экспериментов	credit-scoring-mlflow:latest
postgres	5432	База данных	postgres:15
redis	6379	Кэширование	redis:7-alpine
minio	9000/9001	Объектное хранилище	minio/minio
grafana	3000	Визуализация метрик	grafana/grafana:10.0.0
prometheus	9090	Сбор метрик	prom/prometheus
loki	3100	Хранение логов	grafana/loki
📈 Мониторинг
Evidently AI
Data drift detection: Мониторинг распределения признаков

Concept drift detection: Обнаружение изменений в отношениях признак-таргет

Performance decay monitoring: Контроль качества предсказаний

Графана дашборды
Model Performance Dashboard

Точность, полнота, F1-score в реальном времени

Время ответа модели

Распределение предсказаний

System Health Dashboard

Использование CPU, памяти, диска

Network I/O

Статус сервисов

Data Quality Dashboard

Пропуски в данных

Выбросы

Распределение признаков

Business Metrics Dashboard

Количество обработанных заявок

Процент одобренных кредитов

Средняя сумма кредита

🔄 CI/CD пайплайн
Проект включает GitHub Actions для автоматизации:

Workflows:
test.yml - Автоматическое тестирование

Модульные тесты

Интеграционные тесты

Тесты безопасности (Bandit, Snyk)

build.yml - Сборка Docker образов

Сборка всех сервисов

Push в Container Registry

Security scanning

deploy.yml - Деплой в облако

Terraform apply

Kubernetes deployment

Canary релизы (опционально)

Триггеры:
push в main → тесты → сборка → деплой в production

push в develop → тесты → сборка → деплой в staging

pull request → только тесты

☁️ Деплой в Yandex Cloud
Предварительные требования
bash
# Установите Yandex Cloud CLI
curl https://storage.yandex-cloud.net/yandexcloud-yc/install.sh | bash
yc init

# Установите Terraform
# Для Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y gnupg software-properties-common
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Для MacOS:
brew install terraform
Автоматическое развертывание
bash
# 1. Настройте переменные окружения
export YC_TOKEN=$(yc iam create-token)
export YC_FOLDER_ID=$(yc config get folder-id)
export YC_CLOUD_ID=$(yc config get cloud-id)

# 2. Запустите автоматическое развертывание
make deploy-cloud ENVIRONMENT=production

# ИЛИ по шагам:
make tf-init           # Инициализация Terraform
make tf-plan          # План развертывания
make tf-apply         # Применение конфигурации
make k8s-deploy       # Деплой в Kubernetes
Ручное развертывание
bash
# 1. Создайте сервисный аккаунт
yc iam service-account create --name mlops-sa
yc iam key create --service-account-name mlops-sa --output key.json

# 2. Настройте Terraform
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Отредактируйте terraform.tfvars с вашими значениями

# 3. Создайте инфраструктуру
terraform init
terraform plan
terraform apply -auto-approve

# 4. Настройте доступ к Kubernetes
export CLUSTER_ID=$(terraform output -raw k8s_cluster_id)
yc managed-kubernetes cluster get-credentials $CLUSTER_ID --external

# 5. Деплой приложения
kubectl apply -f kubernetes/
🛠️ Структура проекта
text
Main_prog_ci_cd_ml/
├── configs/                    # Конфигурационные файлы
│   ├── model.yml             # Конфигурация модели
│   ├── api.yml              # Конфигурация API
│   └── monitoring.yml       # Конфигурация мониторинга
├── data/                      # Данные и DVC-конфигурация
│   ├── raw/                 # Сырые данные
│   ├── processed/           # Обработанные данные
│   └── external/            # Внешние данные
├── docker/                    # Dockerfile для сервисов
│   ├── api.Dockerfile       # FastAPI приложение
│   ├── mlflow.Dockerfile    # MLflow
│   └── training.Dockerfile  # Обучение модели
├── infrastructure/           # Terraform-конфигурации
│   ├── main.tf             # Основная конфигурация
│   ├── variables.tf        # Переменные
│   └── outputs.tf          # Выводы
├── kubernetes/              # Манифесты для Kubernetes
│   ├── deployment.yaml     # Деплоймент
│   ├── service.yaml       # Сервис
│   ├── ingress.yaml       # Ingress
│   └── configmap.yaml     # Конфигмапы
├── models/                  # Сохраненные модели
│   ├── trained/           # Обученные модели
│   ├── onnx/             # ONNX модели
│   └── artifacts/         # Артефакты
├── notebooks/              # Jupyter-ноутбуки
│   ├── EDA.ipynb         # Анализ данных
│   ├── Model_Training.ipynb # Обучение модели
│   └── Evaluation.ipynb  # Оценка модели
├── reports/                # Отчеты
│   ├── test_results/     # Результаты тестов
│   ├── benchmarks/       # Бенчмарки
│   └── documentation/    # Документация
├── scripts/               # Вспомогательные скрипты
│   ├── local/           # Локальные скрипты
│   ├── deployment/      # Скрипты деплоя
│   └── monitoring/      # Скрипты мониторинга
├── src/                   # Исходный код
│   ├── api/             # FastAPI приложение
│   ├── ml_pipeline/     # ML пайплайн
│   └── utils/           # Вспомогательные утилиты
├── tests/                # Тесты
│   ├── unit/           # Модульные тесты
│   ├── integration/    # Интеграционные тесты
│   └── e2e/           # End-to-end тесты
├── dags/                 # Airflow DAG для переобучения
├── docker-compose.yml    # Docker Compose конфигурация
├── Makefile             # Управление проектом
├── requirements.txt     # Python зависимости
├── pyproject.toml      # Конфигурация проекта
└── README.md           # Документация


подробнее в папке doc

📋 Полезные команды
Управление проектом через Makefile
bash
# Инициализация и настройка
make init              # Инициализация проекта
make setup             # Настройка окружения
make install           # Установка зависимостей

# Локальная разработка
make run-local         # Запуск всех сервисов локально
make local-status      # Статус локальных сервисов
make local-down        # Остановка локальных сервисов
make pipeline          # Полный ML пайплайн
make api               # Запуск только API
make mlflow            # Запуск только MLflow

# Работа с данными
make data-download     # Загрузка данных
make data-process      # Обработка данных
make data-clean        # Очистка данных

# Обучение модели
make train             # Обучение модели
make evaluate          # Оценка модели
make optimize          # Оптимизация модели
make convert-onnx      # Конвертация в ONNX

# Тестирование
make test              # Запуск всех тестов
make test-unit         # Модульные тесты
make test-integration  # Интеграционные тесты
make test-coverage     # Покрытие тестами

# Мониторинг
make monitor-drift     # Мониторинг дрифта
make monitor-metrics   # Просмотр метрик
make monitor-logs      # Просмотр логов

# Облачное развертывание
make build-all         # Сборка всех Docker образов
make push-images       # Загрузка образов в registry
make tf-init          # Инициализация Terraform
make tf-plan          # План инфраструктуры
make tf-apply         # Создание инфраструктуры
make k8s-deploy       # Деплой в Kubernetes
make k8s-status       # Статус Kubernetes
make deploy-cloud     # Полное облачное развертывание
Работа с Docker
bash
# Запуск и управление
docker-compose up -d              # Запуск всех сервисов
docker-compose down -v            # Остановка с удалением volumes
docker-compose logs -f api        # Просмотр логов API
docker-compose restart api        # Перезапуск API
docker-compose ps                 # Статус контейнеров

# Сборка образов
docker-compose build --no-cache   # Пересборка без кэша
docker-compose build api          # Сборка только API

# Очистка
docker system prune -a            # Очистка Docker
docker volume prune               # Удаление неиспользуемых volumes
Работа с Kubernetes
bash
# Деплой и управление
kubectl apply -f kubernetes/      # Деплой приложения
kubectl get pods -n credit-scoring # Просмотр подов
kubectl get svc -n credit-scoring  # Просмотр сервисов
kubectl get ingress -n credit-scoring # Просмотр ingress

# Логи и отладка
kubectl logs -f deployment/credit-scoring-api -n credit-scoring
kubectl describe pod <pod-name> -n credit-scoring
kubectl exec -it <pod-name> -n credit-scoring -- bash

# Масштабирование
kubectl scale deployment credit-scoring-api --replicas=3 -n credit-scoring
kubectl autoscale deployment credit-scoring-api --min=2 --max=10 --cpu-percent=80 -n credit-scoring

# Port forwarding для локального доступа
kubectl port-forward svc/credit-scoring-api 8000:80 -n credit-scoring
Мониторинг и логи
bash
# Просмотр метрик
curl http://localhost:8000/metrics  # Prometheus метрики API
curl http://localhost:9090          # Prometheus UI

# Проверка здоровья
curl http://localhost:8000/health   # Health check API
curl http://localhost:8000/api/v1/services # Список сервисов

# Логи через Loki
# Используйте Grafana для просмотра логов:
# 1. Откройте http://localhost:3000
# 2. Войдите (admin/admin)
# 3. Перейдите в Explore → Loki
🔐 Безопасность
Управление секретами
Используйте HashiCorp Vault или AWS Secrets Manager для production

Никогда не храните секреты в коде

Используйте переменные окружения через .env файлы

Security scanning
bash
# Сканирование Python кода
bandit -r src/

# Сканирование зависимостей
safety check

# Сканирование Docker образов
trivy image credit-scoring-api:latest

# Сканирование на уязвимости
snyk test
Рекомендации для production
Используйте HTTPS с Let's Encrypt

Настройте WAF (Web Application Firewall)

Реализуйте rate limiting

Настройте мониторинг безопасности

Регулярно обновляйте зависимости

🐛 Устранение неполадок
API не запускается
bash
# Проверьте логи
docker-compose logs api

# Проверьте зависимости
docker-compose exec api pip list

# Проверьте конфигурацию
docker-compose exec api python -c "from src.utils.config import settings; print(settings.dict())"
MLflow не доступен
bash
# Проверьте базу данных
docker-compose exec postgres psql -U postgres -c "\l"

# Проверьте миграции
docker-compose exec mlflow alembic current
Проблемы с данными
bash
# Проверьте доступность MinIO
curl http://localhost:9000/minio/health/live

# Проверьте бакеты
docker-compose exec minio mc ls minio
Проблемы с мониторингом
bash
# Проверьте Prometheus
curl http://localhost:9090/-/healthy

# Проверьте метрики
curl http://localhost:8000/metrics | head -20
```
Скриншоты 

Главный экран
<img width="1124" height="909" alt="image" src="https://github.com/user-attachments/assets/2156d7d2-6b72-4a9e-b408-49fdf5a0b782" />

Модель
<img width="1069" height="930" alt="image" src="https://github.com/user-attachments/assets/194cba5d-3068-414f-aaa0-5075ec826547" />

Графана (пока без граиков но с полной настройкой
<img width="1869" height="414" alt="image" src="https://github.com/user-attachments/assets/6eb3e126-a213-4572-b326-25a386efcf62" />

<img width="1866" height="249" alt="image" src="https://github.com/user-attachments/assets/096a0571-7284-437c-9f1c-cdc23f68d302" />

<img width="1862" height="338" alt="image" src="https://github.com/user-attachments/assets/796bd405-36ab-4ef9-b1a1-5d672fc1f7e7" />
