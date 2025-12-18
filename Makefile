# Makefile для управления проектом MLOps кредитного скоринга
# Поддержка Windows и Linux/macOS

# Определение операционной системы
ifeq ($(OS),Windows_NT)
    # Windows
    SYSTEM := Windows
    VENV_ACTIVATE := . venv\\Scripts\\activate
    VENV_PYTHON := venv\\Scripts\\python
    VENV_PIP := venv\\Scripts\\pip
    RM := del /Q /S
    MKDIR := mkdir
    CP := copy
    SHELL := cmd
else
    # Linux/macOS
    SYSTEM := $(shell uname -s)
    VENV_ACTIVATE := . venv/bin/activate
    VENV_PYTHON := venv/bin/python
    VENV_PIP := venv/bin/pip
    RM := rm -rf
    MKDIR := mkdir -p
    CP := cp
endif

# Цели по умолчанию
.DEFAULT_GOAL := help

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
	@echo "Виртуальное окружение создано"
	@echo ""
	@echo "=== Инструкция ==="
	@echo "1. Активируйте виртуальное окружение:"
ifeq ($(SYSTEM),Windows)
	@echo "   venv\Scripts\activate"
else
	@echo "   source venv/bin/activate"
endif
	@echo "2. Обновите pip:"
	@echo "   python -m pip install --upgrade pip"
	@echo "3. Установите зависимости:"
	@echo "   pip install -r requirements.txt"
	@echo "   pip install -r requirements-dev.txt"
	@echo "4. Установите pre-commit:"
	@echo "   pre-commit install"
	@echo "5. Инициализируйте DVC:"
	@echo "   dvc init"
	@echo ""

# Установка зависимостей
install:
	@echo "Установка зависимостей..."
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PIP) install -r requirements-dev.txt

# Тестирование
test:
	@echo "Запуск тестов..."
	$(VENV_PYTHON) -m pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term-missing
	$(VENV_PYTHON) -m pytest tests/integration/ -v
	$(VENV_PYTHON) -m pytest tests/e2e/ -v

test-unit:
	@echo "Запуск модульных тестов..."
	$(VENV_PYTHON) -m pytest tests/unit/ -v

test-integration:
	@echo "Запуск интеграционных тестов..."
	$(VENV_PYTHON) -m pytest tests/integration/ -v

test-load:
	@echo "Запуск нагрузочного тестов..."
ifeq ($(SYSTEM),Windows)
	@echo "Для Windows используйте:"
	@echo "locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 5m --html=reports/load_test.html"
else
	locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 5m --html=reports/load_test.html
endif

# Проверка кода
lint:
	@echo "Проверка кода..."
	$(VENV_PYTHON) -m flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	$(VENV_PYTHON) -m mypy src/ --ignore-missing-imports
	$(VENV_PYTHON) -m bandit -r src/ -x tests/
	$(VENV_PYTHON) -m safety check -r requirements.txt

format:
	@echo "Форматирование кода..."
	$(VENV_PYTHON) -m black src/ tests/ --line-length=100
	$(VENV_PYTHON) -m isort src/ tests/

# Очистка
clean:
	@echo "Очистка проекта..."
ifeq ($(SYSTEM),Windows)
	@echo "Удаление файлов .pyc..."
	powershell "Get-ChildItem -Path . -Include *.pyc -Recurse | Remove-Item -Force"
	@echo "Удаление __pycache__..."
	powershell "Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Force -Recurse"
	@echo "Удаление .pytest_cache..."
	if exist .pytest_cache rmdir /s /q .pytest_cache
	@echo "Удаление .coverage..."
	if exist .coverage del /q .coverage
	@echo "Удаление htmlcov..."
	if exist htmlcov rmdir /s /q htmlcov
	@echo "Удаление .egg-info..."
	powershell "Get-ChildItem -Path . -Include *.egg-info -Recurse -Directory | Remove-Item -Force -Recurse"
	@echo "Удаление .mypy_cache..."
	if exist .mypy_cache rmdir /s /q .mypy_cache
	@echo "Удаление build, dist, .eggs..."
	if exist build rmdir /s /q build
	if exist dist rmdir /s /q dist
	if exist .eggs rmdir /s /q .eggs
else
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/
endif
	@echo "Очистка завершена"

# Разработка с Docker
up:
	@echo "Запуск сервисов..."
	docker-compose up -d postgres redis minio mlflow
	@echo "Ожидание инициализации сервисов..."
ifeq ($(SYSTEM),Windows)
	timeout /t 10 /nobreak > nul 2>&1 || sleep 10
else
	sleep 10
endif
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
	$(VENV_PYTHON) src/ml_pipeline/training/train_model.py --config configs/training_config.yaml
	@echo "Модель обучена. Результаты в MLflow: http://localhost:5000"

retrain:
	@echo "Переобучение модели..."
	$(VENV_PYTHON) scripts/orchestration/trigger_retraining.py --trigger data_drift
	@echo "Запущено переобучение. Проверьте Airflow: http://localhost:8080"

benchmark:
	@echo "Бенчмаркинг модели..."
	$(VENV_PYTHON) src/ml_pipeline/training/onnx_conversion.py --benchmark
	@echo "Бенчмаркинг завершен. Отчет в reports/benchmark_report.json"

# Инфраструктура
infra-init:
	@echo "Инициализация Terraform..."
	cd infrastructure/environments/staging && terraform init

infra-apply:
	@echo "Развертывание инфраструктуры staging..."
	cd infrastructure/environments/staging && terraform apply -auto-approve
	@echo "Инфраструктура staging развернута"

infra-destroy:
	@echo "Удаление инфраструктуры staging..."
	cd infrastructure/environments/staging && terraform destroy -auto-approve
	@echo "Инфраструктура staging удалена"

# Развертывание
deploy-staging:
	@echo "Развертывание в staging..."
	@echo "Для Windows настройте kubectl вручную:"
	@echo "1. Установите kubectl"
	@echo "2. Настройте kubeconfig"
	@echo "3. Запустите: kubectl apply -f kubernetes/credit-scoring-api/ -n staging"

# Работа с данными
data-download:
	@echo "Загрузка данных..."
	$(VENV_PYTHON) scripts/data/download_data.py --source-url https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
	@echo "Данные загружены в data/raw/"

data-process:
	@echo "Обработка данных..."
	$(VENV_PYTHON) scripts/data/process_data.py --input data/raw/german_credit.csv --output data/processed/ --config configs/processing_config.yaml
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

# Мониторинг и алерты
monitor-drift:
	@echo "Мониторинг дрифта..."
	$(VENV_PYTHON) src/ml_pipeline/monitoring/drift_detection.py --hours 24
	@echo "Мониторинг завершен. Отчет в monitoring/reports/"

# Вспомогательные команды
generate-docs:
	@echo "Генерация документации..."
	$(VENV_PYTHON) -m pdoc --html src --output-dir docs/api --force
	@echo "Документация сгенерирована в docs/api/"

run-notebooks:
	@echo "Запуск Jupyter notebook..."
	docker-compose up jupyter
	@echo "Jupyter доступен по: http://localhost:8888"

backup-models:
	@echo "Создание backup моделей..."
ifeq ($(SYSTEM),Windows)
	powershell "Compress-Archive -Path models -DestinationPath models_backup_$(shell date +%Y%m%d_%H%M%S).zip"
else
	tar -czf models_backup_$$(date +%Y%m%d_%H%M%S).tar.gz models/
endif
	@echo "Backup создан"

restore-models:
	@echo "Восстановление моделей из backup..."
ifeq ($(SYSTEM),Windows)
	@echo "Разархивируйте backup вручную:"
	@echo "powershell Expand-Archive -Path models_backup_*.zip -DestinationPath ."
else
	tar -xzf models_backup_*.tar.gz
endif
	@echo "Модели восстановлены"

# Полный пайплайн
full-pipeline:
	@echo "Запуск полного пайплайна..."
	make clean
	make init
	$(VENV_ACTIVATE) && make install
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
	@echo "Использование диска:"
ifeq ($(SYSTEM),Windows)
	powershell "Get-ChildItem -Path data,models,logs -Directory -ErrorAction SilentlyContinue | Select-Object Name, @{Name='Size(MB)';Expression={[math]::Round((Get-ChildItem $$_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 2)}}"
else
	du -sh data/ models/ logs/ 2>/dev/null || true
endif

# Создание структуры директорий
create-structure:
	@echo "Создание структуры проекта..."
ifeq ($(SYSTEM),Windows)
	@echo "Создание директорий..."
	powershell -Command "mkdir -Force data/raw, data/processed, data/features, models/trained, models/experiments, logs, reports, monitoring/reports, monitoring/alerts, monitoring/dashboard, configs, tests/unit, tests/integration, tests/load, tests/e2e, docker/api, docker/training, docker/monitoring, kubernetes/base, kubernetes/credit-scoring-api, kubernetes/monitoring-stack, kubernetes/ml-pipelines, kubernetes/ingress, infrastructure/modules/network, infrastructure/modules/kubernetes, infrastructure/modules/storage, infrastructure/modules/monitoring, infrastructure/environments/staging, infrastructure/environments/production, scripts/deployment, scripts/monitoring, scripts/benchmarks, docs/runbooks, src/api/routes, src/api/middleware, src/ml_pipeline/training, src/ml_pipeline/inference, src/ml_pipeline/monitoring, src/utils, .github/workflows" 2>nul
	@echo "Создание файлов..."
	echo. > data/.gitkeep
	echo. > models/.gitkeep
	echo. > logs/.gitkeep
	echo. > configs/.gitkeep
else
	mkdir -p data/{raw,processed,features} \
		models/{trained,experiments,champion-challenger} \
		logs \
		reports \
		monitoring/{reports,alerts,dashboard,summaries,retraining} \
		configs \
		tests/{unit,integration,load,e2e} \
		docker/{api,training,monitoring,jupyter} \
		kubernetes/{base,credit-scoring-api,monitoring-stack,ml-pipelines,ingress} \
		infrastructure/modules/{network,kubernetes,storage,monitoring,database} \
		infrastructure/environments/{staging,production} \
		infrastructure/scripts/{deployment,monitoring,benchmarks} \
		scripts/{deployment,monitoring,benchmarks,orchestration,data} \
		docs/runbooks \
		src/api/{routes,middleware,models} \
		src/ml_pipeline/{training,inference,monitoring,validation,registration,deployment} \
		src/utils \
		.github/workflows \
		notebooks
	touch data/.gitkeep models/.gitkeep logs/.gitkeep configs/.gitkeep
endif
	@echo "Структура создана"

# Проверка конфигурации
check-config:
	@echo "Проверка конфигурации..."
	$(VENV_PYTHON) -c "import yaml; config = yaml.safe_load(open('configs/training_config.yaml')); print('training_config keys:', list(config.keys()))"
	@echo ""
	$(VENV_PYTHON) -c "import pandas as pd; df = pd.read_csv('data/processed/train.csv'); print('Колонки в train.csv:', df.columns.tolist())"

# Создание конфигурационных файлов
create-configs:
	@echo "Создание недостающих конфигурационных файлов..."
ifeq ($(SYSTEM),Windows)
	if not exist "configs\processing_config.yaml" \
	( \
		$(VENV_PYTHON) -c "import yaml; config = {'data': {'numerical_features': ['duration', 'credit_amount', 'age', 'installment_commitment', 'residence_since', 'existing_credits', 'num_dependents'], 'categorical_features': ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'], 'target_column': 'default'}, 'model': {'paths': {'scaler': 'models/processed/scaler.pkl', 'encoder': 'models/processed/encoder.pkl'}}}; import os; os.makedirs('configs', exist_ok=True); with open('configs/processing_config.yaml', 'w') as f: yaml.dump(config, f, default_flow_style=False)" \
	)
	if not exist "configs\training_config.yaml" \
	( \
		$(VENV_PYTHON) -c "import yaml; config = {'model_paths': {'trained': 'models/trained/model.pkl', 'onnx': 'models/trained/model.onnx', 'tensorflow': 'models/trained/model'}, 'data': {'train_path': 'data/processed/train.csv', 'test_path': 'data/processed/test.csv', 'target_column': 'default'}}; import os; os.makedirs('configs', exist_ok=True); with open('configs/training_config.yaml', 'w') as f: yaml.dump(config, f, default_flow_style=False)" \
	)
else
	if [ ! -f "configs/processing_config.yaml" ]; then \
		$(VENV_PYTHON) -c "import yaml; config = {'data': {'numerical_features': ['duration', 'credit_amount', 'age', 'installment_commitment', 'residence_since', 'existing_credits', 'num_dependents'], 'categorical_features': ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'], 'target_column': 'default'}, 'model': {'paths': {'scaler': 'models/processed/scaler.pkl', 'encoder': 'models/processed/encoder.pkl'}}}; import os; os.makedirs('configs', exist_ok=True); with open('configs/processing_config.yaml', 'w') as f: yaml.dump(config, f, default_flow_style=False)"; \
	fi
	if [ ! -f "configs/training_config.yaml" ]; then \
		$(VENV_PYTHON) -c "import yaml; config = {'model_paths': {'trained': 'models/trained/model.pkl', 'onnx': 'models/trained/model.onnx', 'tensorflow': 'models/trained/model'}, 'data': {'train_path': 'data/processed/train.csv', 'test_path': 'data/processed/test.csv', 'target_column': 'default'}}; import os; os.makedirs('configs', exist_ok=True); with open('configs/training_config.yaml', 'w') as f: yaml.dump(config, f, default_flow_style=False)"; \
	fi
endif
	@echo "Конфигурационные файлы созданы"

# Быстрая проверка работоспособности
quick-test:
	@echo "Быстрая проверка работоспособности..."
	make data-download
	make create-configs
	make data-process
	make check-config
	@echo "Проверка завершена. Теперь можно запускать: make train"

.PHONY: help init install test test-unit test-integration test-load lint format clean up down logs monitor train retrain benchmark infra-init infra-apply infra-destroy deploy-staging data-download data-process dvc-push build-api build-training build-all ci-test monitor-drift generate-docs run-notebooks backup-models restore-models full-pipeline status create-structure check-config create-configs quick-test