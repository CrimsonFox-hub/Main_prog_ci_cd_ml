# Проект: Промышленное развертывание кредитной скоринговой системы с полным MLOps-циклом

## Описание
Данный проект представляет собой комплексное решение для автоматизации жизненного цикла модели машинного обучения для кредитного скоринга. Реализован полный MLOps-цикл: от подготовки модели и контейнеризации до развертывания в облачной инфраструктуре, настройки CI/CD, мониторинга и автоматического переобучения.

## Ключевые технологии
- **Модель и данные**: PyTorch, ONNX, Scikit-learn, DVC
- **Контейнеризация и оркестрация**: Docker, Docker Compose, Kubernetes
- **Инфраструктура как код**: Terraform (для Yandex Cloud / VK Cloud)
- **CI/CD**: GitHub Actions
- **Мониторинг и логирование**: Prometheus, Grafana, Loki
- **Мониторинг ML-моделей**: Evidently AI
- **Оркестрация пайплайнов**: Apache Airflow

## Структура проекта

``` bash
Main_prog_ci_cd_ml/
├── configs/ # Конфигурационные файлы
├── data/ # Данные и DVC-конфигурация
├── docker/ # Dockerfile для сервисов
├── infrastructure/ # Terraform-конфигурации
├── kubernetes/ # Манифесты для Kubernetes
├── models/ # Сохраненные модели и артефакты
├── notebooks/ # Jupyter-ноутбуки для анализа
├── reports/ # Отчеты по тестированию и бенчмаркам
├── scripts/ # Вспомогательные скрипты
├── src/ # Исходный код
│ ├── api/ # FastAPI-приложение
│ ├── ml_pipeline/ # Пайплайн обучения, инференса, мониторинга
│ └── utils/ # Вспомогательные утилиты
├── tests/ # Модульные и интеграционные тесты
└── dags/ # Airflow DAG для переобучения

```
## Быстрый старт (локально)

## 🚀 Локальный запуск (быстрый старт)

### 1. Предварительные требования
- Docker и Docker Compose
- Git
- Make (опционально, но рекомендуется)

### 2. Настройка проекта
```bash
# Клонирование репозитория
git clone https://github.com/CrimsonFox-hub/Main_prog_ci_cd_ml.git
cd Main_prog_ci_cd_ml

# Настройка окружения (автоматически создаст .env и директории)
make setup-local

# Или вручную:
./scripts/setup-local.sh
# Используя Make (рекомендуется)
make up-all

# Или напрямую:
./scripts/local/start-all.sh

# Для Windows:
make windows-up
# или
.\scripts\windows\start-all.ps1

1. **Клонирование и установка**
    ```bash
    git clone https://github.com/CrimsonFox-hub/Main_prog_ci_cd_ml.git
    cd Main_prog_ci_cd_ml
    python -m venv venv
    # Для Windows: .\venv\Scripts\activate
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

2. **Загрузка данных и обучение модели**
    ```bash
    make data-download
    make data-process
    make train  # Обучает нейронную сеть и экспортирует в ONNX
    ```

3. **Запуск локальных сервисов**
    ```bash
    docker-compose up -d postgres mlflow minio
    make up  # Запуск API и инфраструктуры
    ```
    - API будет доступен на `http://localhost:8000/docs`
    - MLflow UI: `http://localhost:5000`

## Развертывание в облаке

# 1. Установите базовые системные утилиты и Docker с Docker Compose
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common git
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

# 2. Добавьте вашего пользователя в группу docker, чтобы не использовать sudo
```bash
sudo usermod -aG docker $USER

```
# Важно! Для применения изменения нужно выйти и зайти заново в SSH-сессию.
# Сделайте это после завершения всех команд этой части.

# 3. Клонируйте ваш проект с GitHub на ВМ
```bash
git clone https://github.com/CrimsonFox-hub/Main_prog_ci_cd_ml.git
cd Main_prog_ci_cd_ml

# 4. Настройте файл окружения (env). Создайте его на основе примера.
cp .env.example .env
# ОТКРОЙТЕ файл .env в редакторе (nano .env) и заполните реальными данными,
# например, паролями для БД. Для учебного проекта можно оставить упрощенные значения.
```
# установка docker-compose
```bash
apt install docker-compose
```
# 5. Запустите все сервисы
```bash
docker-compose up -d
```
# 6. Проверьте, что все контейнеры работают
```bash
docker-compose ps
```
# 7. Посмотрите логи (особенно если что-то не запустилось)
```bash
docker-compose logs api  # Или имя другого сервиса
```
MinIO логин/пароль:
Логин: minioadmin

Пароль: minioadmin

Откройте: http://localhost:9001

## Мониторинг
После развертывания в облаке настроены дашборды Grafana для мониторинга:
- Метрики инфраструктуры (CPU, память, сеть)
- Метрики производительности модели (латентность, ошибки)
- Детекция дрифта данных и концепта (Evidently AI)

# 1. Создайте ключи доступа для MinIO (если нужно)
docker exec credit_scoring_minio mc admin user svcacct add myminio minioadmin

# 2. Получите список ключей
docker exec credit_scoring_minio mc admin user svcacct list myminio minioadmin

# 3. Удалите старые ключи (если нужно)
docker exec credit_scoring_minio mc admin user svcacct remove myminio minioadmin ACCESS_KEY

# Просмотр логов API
docker-compose logs api

# Просмотр логов MLflow
docker-compose logs mlflow

# Проверка всех логов
docker-compose logs

# Остановка всех сервисов
docker-compose down

# Перезапуск
docker-compose restart api


Сервис	Назначение	URL	Порт
API (Кредитный скоринг)	Основное приложение для прогнозирования	http://localhost:8000	8000
MLflow	Трекинг экспериментов и моделей	http://localhost:5000	5000
MinIO Console	Управление объектным хранилищем	http://localhost:9001	9001
Grafana	Дашборды и визуализация метрик	http://localhost:3000	3000
Loki	Централизованное хранение логов	http://localhost:3100	3100
PostgreSQL (база)	Хранилище данных (используйте клиент, например, DBeaver)	http://localhost:5432	5432
Redis (кэш)	Хранилище ключ-значение (используйте клиент, например, RedisInsight)	http://localhost:6379	6379


# Credit Scoring MLOps System

Полная промышленная система для кредитного скоринга с использованием нейронных сетей и MLOps практик.

## 🎯 Особенности

- **Нейронная сеть** на PyTorch с архитектурой 128-64-32
- **ONNX формат** для production инференса
- **Quantization & Pruning** для оптимизации
- **Полный MLOps стек**: MLflow, Grafana, Prometheus, Loki
- **Docker Compose** для оркестрации
- **FastAPI** с веб-интерфейсом
- **Автоматический мониторинг** дрифта данных

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Клонирование репозитория
git clone <repository-url>
cd credit-scoring-mlops

# Установка Python зависимостей
pip install -r requirements.txt

# Установка Docker и Docker Compose
# См. инструкции на docker.com
2. Запуск всей системы
bash
# Запуск всех компонентов одним скриптом
python run_all.py
Система автоматически:

Запустит все Docker сервисы

Обучит модель (если не обучена)

Конвертирует модель в ONNX

Запустит FastAPI сервер

Откроет веб-интерфейс в браузере

3. Ручной запуск
bash
# 1. Запуск Docker сервисов
docker-compose up -d

# 2. Обучение модели
python -m src.ml_pipeline.training.train_model

# 3. Конвертация в ONNX
python src/ml_pipeline/training/onnx_conversion.py

# 4. Запуск API
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
🌐 Веб-интерфейс
После запуска откройте в браузере:

Сервис	Назначение	URL
Главная страница	Демонстрация системы	http://localhost:8000
Демо модель	Тестирование нейронной сети	http://localhost:8000/demo
API документация	OpenAPI/Swagger UI	http://localhost:8000/api/docs
MLflow	Трекинг экспериментов	http://localhost:5000
Grafana	Дашборды и визуализация	http://localhost:3000
MinIO Console	Объектное хранилище	http://localhost:9001
Loki	Просмотр логов	http://localhost:3100
Данные для входа:

Grafana: admin/admin

MinIO: minioadmin/minioadmin

🧠 Архитектура модели
Нейронная сеть
python
CreditScoringNN(
    input_size=20,  # Количество признаков
    hidden_layers=[128, 64, 32],
    dropout_rate=0.3
)
Оптимизации
Pruning: Удаление 20% наименее значимых весов

Quantization: 8-bit динамическое квантование

ONNX: Оптимизированный формат для production

Метрики производительности
Точность: ~85%

Время инференса: <5 мс

Размер модели: <2 MB

📊 Модели данных
Запрос для предсказания
json
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
Ответ
json
{
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "low",
  "recommendation": "Кредит рекомендуется к одобрению",
  "explanation": {
    "top_features": ["credit_score", "income", "debt_to_income"]
  }
}
🔧 API эндпоинты
Основные
POST /api/v1/predict - Предсказание для одного клиента

POST /api/v1/batch_predict - Пакетное предсказание

GET /health - Проверка здоровья системы

GET /api/v1/models/info - Информация о модели

Мониторинг
GET /api/v1/system/metrics - Метрики для Prometheus

GET /api/v1/services - Список сервисов

🐳 Docker сервисы
Сервис	Порт	Описание
API	8000	FastAPI приложение
MLflow	5000	Трекинг экспериментов
PostgreSQL	5432	Основная база данных
Redis	6379	Кэширование
MinIO	9000/9001	Объектное хранилище
Grafana	3000	Визуализация метрик
Prometheus	9090	Сбор метрик
Loki	3100	Хранение логов
📈 Мониторинг
Evidently AI
Мониторинг дрифта данных и концепта:

Data drift detection

Concept drift detection

Performance decay monitoring

Графана дашборды
Model Performance - Метрики модели

System Health - Мониторинг инфраструктуры

Data Quality - Качество входных данных

Business Metrics - Бизнес-показатели

🔄 CI/CD пайплайн
Проект включает GitHub Actions для:

Автоматического тестирования

Security scanning

Деплоя в staging/production

Canary релизов (опционально)

🚢 Деплой в Yandex Cloud
1. Настройка инфраструктуры
bash
cd terraform/
terraform init
terraform plan
terraform apply
2. Настройка Kubernetes
bash
kubectl apply -f k8s/
3. Настройка домена
Добавить A-записи для:

api.your-domain.com → API

mlflow.your-domain.com → MLflow

grafana.your-domain.com → Grafana

📚 Полезные команды
Обучение и оптимизация
bash
# Обучение модели
python -m src.ml_pipeline.training.train_model

# Конвертация в ONNX
python src/ml_pipeline/training/onnx_conversion.py

# Оптимизация модели
python src/ml_pipeline/optimization/model_optimizer.py

# Бенчмаркинг
python src/ml_pipeline/optimization/benchmark.py
Мониторинг
bash
# Мониторинг дрифта
python src/ml_pipeline/monitoring/drift_detection.py

# Запуск Evidently дашбордов
python src/ml_pipeline/monitoring/evidently_dashboard.py
Управление Docker
bash
# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f api

# Остановка
docker-compose down -v

# Пересборка
docker-compose up -d --build
Kubernetes
bash
# Деплой
kubectl apply -f k8s/

# Просмотр подов
kubectl get pods -n credit-scoring

# Просмотр логов
kubectl logs -f deployment/credit-scoring-api -n credit-scoring

# Port forwarding
kubectl port-forward svc/credit-scoring-api 8000:8000 -n credit-scoring
🛠️ Структура проекта
text
credit-scoring-mlops/
├── src/                          # Исходный код
│   ├── api/                      # FastAPI приложение
│   │   ├── app.py               # Основное приложение
│   │   ├── static/              # Статические файлы
│   │   └── templates/           # HTML шаблоны
│   ├── ml_pipeline/             # ML пайплайн
│   │   ├── training/            # Обучение модели
│   │   ├── inference/           # Инференс
│   │   ├── optimization/        # Оптимизация
│   │   └── monitoring/          # Мониторинг
│   └── utils/                   # Утилиты
├── models/                       # Модели
├── data/                         # Данные
├── configs/                      # Конфигурации
├── k8s/                         # Kubernetes манифесты
├── terraform/                   # Infrastructure as Code
├── docker-compose.yml           # Docker Compose
├── requirements.txt             # Python зависимости
├── run_all.py                  # Скрипт запуска
└── README.md                   # Документация
🔐 Безопасность
Secrets management
Используйте HashiCorp Vault или AWS Secrets Manager

Никогда не храните секреты в коде

Используйте переменные окружения

Security scanning
bash
# Snyk для сканирования уязвимостей
snyk test

# Bandit для анализа Python кода
bandit -r src/

Начните использовать Credit Scoring System прямо сейчас!

bash
# Клонируйте и запустите
git clone <repository-url>
cd credit-scoring-mlops
python run_all.py
Откройте http://localhost:8000 и начните тестирование!

text

## Запуск проекта

1. **Установите зависимости:**
```bash
pip install -r requirements.txt
Запустите всю систему:

bash
python run_all.py
Или запустите вручную:

bash
# Запуск Docker сервисов
docker-compose up -d

# Обучение модели
python -m src.ml_pipeline.training.train_model

# Конвертация в ONNX
python src/ml_pipeline/training/onnx_conversion.py

# Запуск API
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
Демонстрация
После запуска откройте в браузере:

Главная страница: http://localhost:8000

Демо модель: http://localhost:8000/demo

Все сервисы: http://localhost:8000/services

Сервисы
Все сервисы будут доступны по адресам:

Сервис	URL	Порт
API	http://localhost:8000	8000
MLflow	http://localhost:5000	5000
Grafana	http://localhost:3000	3000
MinIO	http://localhost:9001	9001
Loki	http://localhost:3100	3100
Prometheus	http://localhost:9090	9090
PostgreSQL	localhost:5432	5432
Redis	localhost:6379	6379