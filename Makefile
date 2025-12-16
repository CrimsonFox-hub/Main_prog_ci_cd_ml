# Makefile для управления проектом MLOps кредитного скоринга

.PHONY: help init install test lint format clean build deploy infra up down logs monitor train retrain benchmark

# Цели по умолчанию
help:
	@echo "MLOps Credit Scoring Project"
	@echo ""
	@echo "Основные команды:"
	@echo "  make init               - Инициализация проекта"
	@echo "  make install            - Установка зависимостей"
	@echo "  make test               - Запуск всех тестов"
	@echo "  make lint               - Проверка кода"
	@echo "  make format             - Форматирование кода"
	@echo "  make clean              - Очистка проекта"
	@echo ""
	@echo "Разработка:"
	@echo "  make up                 - Запуск всех сервисов"
	@echo "  make down               - Остановка всех сервисов"
	@echo "  make logs               - Просмотр логов"
	@echo "  make monitor            - Открыть мониторинг"
	@echo ""
	@echo "Модели:"
	@echo "  make train              - Обучение модели"
	@echo "  make retrain            - Переобучение модели"
	@echo "  make benchmark          - Бенчмаркинг модели"
	@echo ""
	@echo "Инфраструктура:"
	@echo "  make infra-apply        - Развертывание инфраструктуры"
	@echo "  make infra-destroy      - Удаление инфраструктуры"
	@echo "  make deploy-staging     - Развертывание в staging"
	@echo "  make deploy-production  - Развертывание в production"
	@echo ""
	@echo "Данные:"
	@echo "  make data-download      - Загрузка данных"
	@echo "  make data-process       - Обработка данных"
	@echo "  make dvc-push           - Сохранение данных в DVC"

# Инициализация проекта
init:
	@echo "Инициализация проекта..."
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements-dev.txt
	pre-commit install
	dvc init
	@echo "Проект инициализирован. Активируйте виртуальное окружение: source venv/bin/activate"

# Установка зависимостей
install:
	@echo "Установка зависимостей..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Тестирование
test:
	@echo "Запуск тестов..."
	pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term-missing
	pytest tests/integration/ -v
	pytest tests/e2e/ -v

test-unit:
	@echo "Запуск модульных тестов..."
	pytest tests/unit/ -v

test-integration:
	@echo "Запуск интеграционных тестов..."
	pytest tests/integration/ -v

test-load:
	@echo "Запуск нагрузочного тестов..."
	locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 5m --html=reports/load_test.html

# Проверка кода
lint:
	@echo "Проверка кода..."
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	mypy src/ --ignore-missing-imports
	bandit -r src/ -x tests/
	safety check -r requirements.txt

format:
	@echo "Форматирование кода..."
	black src/ tests/ --line-length=100
	isort src/ tests/

# Очистка
clean:
	@echo "Очистка проекта..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/
	@echo "Очистка завершена"

# Разработка с Docker
up:
	@echo "Запуск сервисов..."
	docker-compose up -d postgres redis minio mlflow
	@echo "Ожидание инициализации сервисов..."
	sleep 10
	docker-compose up -d api
	@echo "Сервисы запущены"
	@echo "API: http://localhost:8000"
	@echo "MLflow: http://localhost:5000"
	@echo "MinIO: http://localhost:9001"
	@echo "Документация API: http://localhost:8000/docs"

down:
	@echo "Остановка сервисов..."
	docker-compose down
	@echo "Сервисы остановлены"

logs:
	@echo "Просмотр логов..."
	docker-compose logs -f api

monitor:
	@echo "Запуск мониторинга..."
	docker-compose up -d prometheus grafana
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

# Работа с моделями
train:
	@echo "Обучение модели..."
	python src/ml_pipeline/training/train_model.py --config configs/training_config.yaml
	@echo "Модель обучена. Результаты в MLflow: http://localhost:5000"

retrain:
	@echo "Переобучение модели..."
	python scripts/orchestration/trigger_retraining.py --trigger data_drift
	@echo "Запущено переобучение. Проверьте Airflow: http://localhost:8080"

benchmark:
	@echo "Бенчмаркинг модели..."
	python src/ml_pipeline/training/onnx_conversion.py --benchmark
	@echo "Бенчмаркинг завершен. Отчет в reports/benchmark_report.json"

# Инфраструктура
infra-init:
	@echo "Инициализация Terraform..."
	cd infrastructure/environments/staging && terraform init

infra-apply:
	@echo "Развертывание инфраструктуры staging..."
	cd infrastructure/environments/staging && terraform apply -auto-approve
	@echo "Инфраструктура staging развернута"

infra-apply-prod:
	@echo "Развертывание инфраструктуры production..."
	cd infrastructure/environments/production && terraform apply -auto-approve
	@echo "Инфраструктура production развернута"

infra-destroy:
	@echo "Удаление инфраструктуры staging..."
	cd infrastructure/environments/staging && terraform destroy -auto-approve
	@echo "Инфраструктура staging удалена"

infra-destroy-prod:
	@echo "Удаление инфраструктуры production..."
	cd infrastructure/environments/production && terraform destroy -auto-approve
	@echo "Инфраструктура production удалена"

# Развертывание
deploy-staging:
	@echo "Развертывание в staging..."
	export KUBECONFIG=$$(pwd)/kubeconfig-staging.yaml
	kubectl apply -f kubernetes/base/ -n staging
	kubectl apply -f kubernetes/credit-scoring-api/ -n staging
	kubectl rollout status deployment/credit-scoring-api -n staging --timeout=5m
	@echo "Развертывание в staging завершено"
	@echo "Сервис доступен по: $$(kubectl get svc credit-scoring-api -n staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"

deploy-production:
	@echo "Развертывание в production..."
	export KUBECONFIG=$$(pwd)/kubeconfig-production.yaml
	kubectl apply -f kubernetes/base/ -n production
	kubectl apply -f kubernetes/credit-scoring-api/ -n production
	kubectl rollout status deployment/credit-scoring-api -n production --timeout=5m
	@echo "Развертывание в production завершено"
	@echo "Сервис доступен по: $$(kubectl get svc credit-scoring-api -n production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"

# Работа с данными
data-download:
	@echo "Загрузка данных..."
	python scripts/data/download_data.py --source-url https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
	@echo "Данные загружены в data/raw/"

data-process:
	@echo "Обработка данных..."
	python scripts/data/process_data.py --input data/raw/german.data --output data/processed/
	@echo "Данные обработаны и сохранены в data/processed/"

dvc-push:
	@echo "Сохранение данных в DVC..."
	dvc add data/processed/train.csv data/processed/test.csv
	dvc push
	git add data/processed/.gitignore data/processed/train.csv.dvc data/processed/test.csv.dvc
	@echo "Данные сохранены в DVC"

# Сборка Docker образов
build-api:
	@echo "Сборка Docker образа API..."
	docker build -t credit-scoring-api:latest -f docker/api/Dockerfile .
	@echo "Образ собран: credit-scoring-api:latest"

build-training:
	@echo "Сборка Docker образа для обучения..."
	docker build -t credit-scoring-training:latest -f docker/training/Dockerfile .
	@echo "Образ собран: credit-scoring-training:latest"

build-all:
	@echo "Сборка всех Docker образов..."
	make build-api
	make build-training
	docker build -t credit-scoring-monitoring:latest -f docker/monitoring/Dockerfile .
	@echo "Все образы собраны"

# CI/CD
ci-test:
	@echo "Запуск CI пайплайна..."
	make lint
	make test-unit
	make test-integration
	@echo "CI пайплайн завершен успешно"

cd-deploy:
	@echo "Запуск CD пайплайна..."
	make build-all
	make deploy-staging
	make test-load
	@echo "CD пайплайн завершен успешно"

# Мониторинг и алерты
monitor-drift:
	@echo "Мониторинг дрифта..."
	python src/ml_pipeline/monitoring/drift_detection.py --hours 24
	@echo "Мониторинг завершен. Отчет в monitoring/reports/"

setup-alerts:
	@echo "Настройка алертов..."
	kubectl apply -f kubernetes/monitoring-stack/alertmanager/
	@echo "Алерты настроены"

# Вспомогательные команды
generate-docs:
	@echo "Генерация документации..."
	pdoc --html src --output-dir docs/api --force
	@echo "Документация сгенерирована в docs/api/"

run-notebooks:
	@echo "Запуск Jupyter notebook..."
	docker-compose up jupyter
	@echo "Jupyter доступен по: http://localhost:8888"

backup-models:
	@echo "Создание backup моделей..."
	tar -czf models_backup_$$(date +%Y%m%d_%H%M%S).tar.gz models/
	@echo "Backup создан"

restore-models:
	@echo "Восстановление моделей из backup..."
	tar -xzf models_backup_*.tar.gz
	@echo "Модели восстановлены"

# Полный пайплайн
full-pipeline:
	@echo "Запуск полного пайплайна..."
	make clean
	make init
	make data-download
	make data-process
	make train
	make benchmark
	make test
	make build-all
	@echo "Полный пайплайн завершен"

# Статус сервисов
status:
	@echo "Статус сервисов:"
	docker-compose ps
	@echo ""
	@echo "Kubernetes deployments:"
	kubectl get deployments --all-namespaces 2>/dev/null || echo "Kubernetes не настроен"
	@echo ""
	@echo "Использование диска:"
	du -sh data/ models/ logs/ 2>/dev/null || true

# Помощь с настройкой
setup-help:
	@echo "=== Настройка проекта ==="
	@echo ""
	@echo "1. Клонируйте репозиторий:"
	@echo "   git clone <repo-url>"
	@echo "   cd mlops-credit-scoring"
	@echo ""
	@echo "2. Инициализируйте проект:"
	@echo "   make init"
	@echo "   source venv/bin/activate"
	@echo ""
	@echo "3. Настройте переменные окружения:"
	@echo "   cp .env.example .env"
	@echo "   # Отредактируйте .env файл"
	@echo ""
	@echo "4. Загрузите и обработайте данные:"
	@echo "   make data-download"
	@echo "   make data-process"
	@echo ""
	@echo "5. Запустите сервисы для разработки:"
	@echo "   make up"
	@echo ""
	@echo "6. Обучите модель:"
	@echo "   make train"
	@echo ""
	@echo "7. Запустите тесты:"
	@echo "   make test"
	@echo ""
	@echo "8. Разверните в staging:"
	@echo "   make infra-apply"
	@echo "   make deploy-staging"
	@echo ""
	@echo "Документация API: http://localhost:8000/docs"
	@echo "MLflow: http://localhost:5000"
	@echo "Grafana: http://localhost:3000"