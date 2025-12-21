# ============================================================================
# CREDIT SCORING MLOPS SYSTEM - MAKE FILE
# ============================================================================

# Конфигурация проекта
PROJECT_NAME := credit-scoring-mlops
VERSION := 2.0.0
ENVIRONMENT ?= local

# Цвета для вывода
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m # No Color

# Пути к Docker файлам
DOCKER_API := docker/api/Dockerfile
DOCKER_TRAINING := docker/training/Dockerfile
DOCKER_MLFLOW := docker/mlflow/Dockerfile
DOCKER_MONITORING := docker/monitoring/Dockerfile

# Имена Docker образов
IMAGE_API := $(PROJECT_NAME)-api:$(VERSION)
IMAGE_TRAINING := $(PROJECT_NAME)-training:$(VERSION)
IMAGE_MLFLOW := $(PROJECT_NAME)-mlflow:$(VERSION)
IMAGE_MONITORING := $(PROJECT_NAME)-monitoring:$(VERSION)

# Конфигурации для разных сред
ifeq ($(ENVIRONMENT), local)
	DOCKER_REGISTRY := localhost:5000
	DOCKER_COMPOSE_FILE := docker-compose.local.yml
else ifeq ($(ENVIRONMENT), staging)
	DOCKER_REGISTRY := cr.yandex.cloud/$(YC_REGISTRY_ID)
	DOCKER_COMPOSE_FILE := docker-compose.staging.yml
else ifeq ($(ENVIRONMENT), production)
	DOCKER_REGISTRY := cr.yandex.cloud/$(YC_REGISTRY_ID)
	DOCKER_COMPOSE_FILE := docker-compose.prod.yml
endif

# Kubernetes конфигурация
K8S_NAMESPACE := credit-scoring
K8S_CONTEXT := yc-$(K8S_NAMESPACE)
K8S_DIR := k8s/manifests

# Terraform конфигурация
TF_DIR := terraform
TF_STATE := terraform.tfstate
TF_VARS := terraform.tfvars

# Директории
DATA_DIR := data
MODELS_DIR := models
REPORTS_DIR := reports
LOGS_DIR := logs
MLRUNS_DIR := mlruns

# Python
PYTHON := python3
PIP := pip3
PYTEST := pytest

# ============================================================================
# ПОМОЩЬ И ИНФОРМАЦИЯ
# ============================================================================

.PHONY: help info version

help: ## Показать эту справку
	@echo "$(CYAN)╔════════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║           CREDIT SCORING MLOPS SYSTEM - СИСТЕМА УПРАВЛЕНИЯ            ║$(NC)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Использование:$(NC)"
	@echo "  make [цель] ENVIRONMENT=[local|staging|production]"
	@echo ""
	@echo "$(YELLOW)Основные цели:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Примеры:$(NC)"
	@echo "  make build-all ENVIRONMENT=local      # Собрать все образы для локальной среды"
	@echo "  make deploy-local                     # Локальное развертывание"
	@echo "  make deploy-yc                        # Развертывание в Yandex Cloud"
	@echo "  make train                            # Обучение модели"
	@echo ""

info: ## Информация о текущей конфигурации
	@echo "$(YELLOW)Конфигурация проекта:$(NC)"
	@echo "  Проект:          $(PROJECT_NAME) v$(VERSION)"
	@echo "  Среда:           $(ENVIRONMENT)"
	@echo "  Docker Registry: $(DOCKER_REGISTRY)"
	@echo "  Kubernetes:      $(K8S_NAMESPACE) ($(K8S_CONTEXT))"
	@echo ""
	@echo "$(YELLOW)Docker образы:$(NC)"
	@echo "  API:          $(IMAGE_API)"
	@echo "  Training:     $(IMAGE_TRAINING)"
	@echo "  MLflow:       $(IMAGE_MLFLOW)"
	@echo "  Monitoring:   $(IMAGE_MONITORING)"
	@echo ""
	@echo "$(YELLOW)Директории:$(NC)"
	@echo "  Terraform:    $(TF_DIR)"
	@echo "  Kubernetes:   $(K8S_DIR)"
	@echo "  Модели:       $(MODELS_DIR)"
	@echo "  Данные:       $(DATA_DIR)"

version: ## Показать версию системы
	@echo "$(PROJECT_NAME) v$(VERSION)"


# Yandex Cloud Deploy
yc-setup:
	@echo "Установка yc CLI..."
	curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash

yc-auth:
	@echo "Аутентификация в Yandex Cloud..."
	yc init

yc-deploy:
	@echo "Деплой в Yandex Cloud..."
	chmod +x scripts/yandex-cloud/deploy-vm.sh
	./scripts/yandex-cloud/deploy-vm.sh

yc-destroy:
	@echo "Удаление ресурсов Yandex Cloud..."
	yc compute instance delete mlops-credit-scoring

yc-status:
	@echo "Статус VM в Yandex Cloud..."
	yc compute instance list

# Общие команды
fix-imports:
	@echo "Создание недостающих файлов..."
	python scripts/create_missing_files.py

test-local:
	@echo "Тестирование локального запуска..."
	docker-compose down
	docker-compose build
	docker-compose up -d

	curl http://localhost:8000/health

deploy-all: fix-imports test-local yc-deploy
	@echo "Полный деплой завершен!"

.PHONY: yc-setup yc-auth yc-deploy yc-destroy yc-status fix-imports test-local deploy-all
# ============================================================================
# УСТАНОВКА И НАСТРОЙКА
# ============================================================================

.PHONY: init setup check-deps check-docker check-k8s check-yc

init: ## Инициализация проекта (создание директорий)
	@echo "$(GREEN)Инициализация проекта...$(NC)"
	@mkdir -p $(DATA_DIR)/{raw,processed,external}
	@mkdir -p $(MODELS_DIR)/{processed,deployed}
	@mkdir -p $(REPORTS_DIR)/{benchmarks,drift,performance}
	@mkdir -p $(LOGS_DIR)/{api,training,inference}
	@mkdir -p $(MLRUNS_DIR)
	@mkdir -p $(K8S_DIR)/{configs,secrets,deployments,services,ingress}
	@echo "$(GREEN)✓ Директории созданы$(NC)"

setup: check-deps ## Установка Python зависимостей
	@echo "$(GREEN)Установка зависимостей...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Зависимости установлены$(NC)"

check-deps: ## Проверка зависимостей
	@echo "$(GREEN)Проверка зависимостей...$(NC)"
	@which $(PYTHON) >/dev/null || (echo "$(RED)❌ Python3 не установлен$(NC)" && exit 1)
	@which $(PIP) >/dev/null || (echo "$(RED)❌ pip3 не установлен$(NC)" && exit 1)
	@which docker >/dev/null || (echo "$(YELLOW)⚠️  Docker не установлен$(NC)")
	@which docker-compose >/dev/null || (echo "$(YELLOW)⚠️  Docker Compose не установлен$(NC)")
	@echo "$(GREEN)✓ Зависимости проверены$(NC)"

check-docker: ## Проверка Docker
	@echo "$(GREEN)Проверка Docker...$(NC)"
	@docker --version || (echo "$(RED)❌ Docker не установлен или не запущен$(NC)" && exit 1)
	@docker-compose --version || (echo "$(RED)❌ Docker Compose не установлен$(NC)" && exit 1)
	@echo "$(GREEN)✓ Docker проверен$(NC)"

check-k8s: ## Проверка Kubernetes
	@echo "$(GREEN)Проверка Kubernetes...$(NC)"
	@which kubectl >/dev/null || (echo "$(RED)❌ kubectl не установлен$(NC)" && exit 1)
	@kubectl version --client >/dev/null || (echo "$(RED)❌ kubectl не настроен$(NC)" && exit 1)
	@echo "$(GREEN)✓ Kubernetes проверен$(NC)"

check-yc: ## Проверка Yandex Cloud CLI
	@echo "$(GREEN)Проверка Yandex Cloud...$(NC)"
	@which yc >/dev/null || (echo "$(RED)❌ Yandex Cloud CLI не установлен$(NC)" && exit 1)
	@yc config list >/dev/null || (echo "$(RED)❌ Yandex Cloud не настроен$(NC)" && exit 1)
	@echo "$(GREEN)✓ Yandex Cloud проверен$(NC)"

# ============================================================================
# DOCKER СБОРКА
# ============================================================================

.PHONY: build build-api build-training build-mlflow build-monitoring build-all

build-api: check-docker ## Сборка API образа
	@echo "$(GREEN)Сборка API образа...$(NC)"
	docker build -f $(DOCKER_API) -t $(IMAGE_API) .
	@echo "$(GREEN)✓ API образ собран: $(IMAGE_API)$(NC)"

build-training: check-docker ## Сборка Training образа
	@echo "$(GREEN)Сборка Training образа...$(NC)"
	docker build -f $(DOCKER_TRAINING) -t $(IMAGE_TRAINING) .
	@echo "$(GREEN)✓ Training образ собран: $(IMAGE_TRAINING)$(NC)"

build-mlflow: check-docker ## Сборка MLflow образа
	@echo "$(GREEN)Сборка MLflow образа...$(NC)"
	docker build -f $(DOCKER_MLFLOW) -t $(IMAGE_MLFLOW) .
	@echo "$(GREEN)✓ MLflow образ собран: $(IMAGE_MLFLOW)$(NC)"

build-monitoring: check-docker ## Сборка Monitoring образа
	@echo "$(GREEN)Сборка Monitoring образа...$(NC)"
	docker build -f $(DOCKER_MONITORING) -t $(IMAGE_MONITORING) .
	@echo "$(GREEN)✓ Monitoring образ собран: $(IMAGE_MONITORING)$(NC)"

build-all: build-api build-training build-mlflow build-monitoring ## Сборка всех образов
	@echo "$(GREEN)✓ Все Docker образы собраны$(NC)"

# ============================================================================
# ЛОКАЛЬНОЕ РАЗВЕРТЫВАНИЕ
# ============================================================================

.PHONY: local-up local-down local-logs local-status local-restart

local-up: build-all check-docker ## Локальный запуск (Docker Compose)
	@echo "$(GREEN)Запуск локального окружения...$(NC)"
	docker-compose -f docker-compose.local.yml up -d
	@echo "$(GREEN)✓ Локальное окружение запущено$(NC)"
	@echo ""
	@echo "$(CYAN)СЕРВИСЫ ДОСТУПНЫ:$(NC)"
	@echo "  API:          http://localhost:8000"
	@echo "  API Docs:     http://localhost:8000/api/docs"
	@echo "  MLflow:       http://localhost:5000"
	@echo "  MinIO:        http://localhost:9001"
	@echo "  Grafana:      http://localhost:3000"
	@echo "  Prometheus:   http://localhost:9090"
	@echo "  Loki:         http://localhost:3100"
	@echo ""
	@echo "Для остановки выполните: $(YELLOW)make local-down$(NC)"

local-down: ## Остановка локального окружения
	@echo "$(YELLOW)Остановка локального окружения...$(NC)"
	docker-compose -f docker-compose.local.yml down
	@echo "$(GREEN)✓ Локальное окружение остановлено$(NC)"

local-logs: ## Просмотр логов локального окружения
	@echo "$(GREEN)Просмотр логов...$(NC)"
	docker-compose -f docker-compose.local.yml logs -f

local-status: ## Статус локальных контейнеров
	@echo "$(GREEN)Статус контейнеров:$(NC)"
	docker-compose -f docker-compose.local.yml ps

local-restart: local-down local-up ## Перезапуск локального окружения

# ============================================================================
# ML ПАЙПЛАЙН
# ============================================================================

.PHONY: train convert optimize benchmark pipeline retrain

train: ## Обучение нейронной сети
	@echo "$(GREEN)Обучение нейронной сети...$(NC)"
	$(PYTHON) main_pipeline.py --mode train --config configs/training_config.yaml
	@echo "$(GREEN)✓ Модель обучена$(NC)"

convert: ## Конвертация модели в ONNX
	@echo "$(GREEN)Конвертация модели в ONNX...$(NC)"
	$(PYTHON) src/ml_pipeline/training/onnx_conversion.py \
		--model-path $(MODELS_DIR)/credit_scoring.pth \
		--input-size 20 \
		--output-path $(MODELS_DIR)/credit_scoring.onnx
	@echo "$(GREEN)✓ Модель сконвертирована в ONNX$(NC)"

optimize: ## Оптимизация модели (Quantization + Pruning)
	@echo "$(GREEN)Оптимизация модели...$(NC)"
	$(PYTHON) src/ml_pipeline/optimization/model_optimizer.py \
		--model-path $(MODELS_DIR)/credit_scoring.onnx \
		--input-size 20 \
		--quantize \
		--prune \
		--output-path $(MODELS_DIR)/credit_scoring_optimized.onnx
	@echo "$(GREEN)✓ Модель оптимизирована$(NC)"

benchmark: ## Бенчмаркинг моделей
	@echo "$(GREEN)Запуск бенчмарка моделей...$(NC)"
	$(PYTHON) src/ml_pipeline/optimization/benchmark.py \
		--models $(MODELS_DIR)/credit_scoring.onnx $(MODELS_DIR)/credit_scoring_optimized.onnx \
		--output $(REPORTS_DIR)/benchmarks/benchmark_results.json
	@echo "$(GREEN)✓ Бенчмарк завершен$(NC)"

pipeline: train convert optimize benchmark ## Полный ML пайплайн
	@echo "$(GREEN)✓ Полный ML пайплайн выполнен$(NC)"

retrain: ## Переобучение модели (Airflow DAG)
	@echo "$(GREEN)Запуск переобучения модели...$(NC)"
	docker exec -it airflow-worker airflow dags trigger credit_scoring_retraining_pipeline
	@echo "$(GREEN)✓ Переобучение запущено$(NC)"

# ============================================================================
# ТЕСТИРОВАНИЕ И ВАЛИДАЦИЯ
# ============================================================================

.PHONY: test test-unit test-integration test-api test-model validate

test: ## Запуск всех тестов
	@echo "$(GREEN)Запуск всех тестов...$(NC)"
	$(PYTEST) tests/ -v

test-unit: ## Юнит-тесты
	@echo "$(GREEN)Юнит-тесты...$(NC)"
	$(PYTEST) tests/unit/ -v

test-integration: ## Интеграционные тесты
	@echo "$(GREEN)Интеграционные тесты...$(NC)"
	$(PYTEST) tests/integration/ -v

test-api: ## Тестирование API
	@echo "$(GREEN)Тестирование API...$(NC)"
	$(PYTEST) tests/api/ -v

test-model: ## Тестирование модели
	@echo "$(GREEN)Тестирование модели...$(NC)"
	$(PYTHON) scripts/test_model.py \
		--model $(MODELS_DIR)/credit_scoring.onnx \
		--test-data $(DATA_DIR)/processed/test.csv

validate: ## Валидация пайплайна
	@echo "$(GREEN)Валидация пайплайна...$(NC)"
	$(PYTHON) scripts/validate_pipeline.py --config configs/validation_config.yaml

# ============================================================================
# НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ
# ============================================================================

.PHONY: load-test stress-test performance-test

load-test: ## Нагрузочное тестирование API
	@echo "$(GREEN)Нагрузочное тестирование API...$(NC)"
	$(PYTHON) scripts/load_test.py \
		--url http://localhost:8000/api/v1/predict \
		--rate 100 \
		--duration 60 \
		--output $(REPORTS_DIR)/performance/load_test.json

stress-test: ## Стресс-тестирование
	@echo "$(GREEN)Стресс-тестирование...$(NC)"
	$(PYTHON) scripts/stress_test.py \
		--model $(MODELS_DIR)/credit_scoring.onnx \
		--duration 300 \
		--concurrency 50

performance-test: load-test stress-test ## Все тесты производительности

# ============================================================================
# МОНИТОРИНГ И OBSERVABILITY
# ============================================================================

.PHONY: monitor monitor-drift monitor-metrics monitor-logs dashboard

monitor: ## Запуск мониторинга
	@echo "$(GREEN)Запуск системы мониторинга...$(NC)"
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "$(GREEN)✓ Мониторинг запущен$(NC)"

monitor-drift: ## Мониторинг дрифта данных
	@echo "$(GREEN)Мониторинг дрифта данных...$(NC)"
	$(PYTHON) src/ml_pipeline/monitoring/drift_detection.py \
		--config configs/monitoring_config.yaml \
		--hours 24 \
		--output $(REPORTS_DIR)/drift/drift_report.html

monitor-metrics: ## Просмотр метрик
	@echo "$(GREEN)Метрики системы:$(NC)"
	@curl -s http://localhost:8000/api/v1/system/metrics | python -m json.tool

monitor-logs: ## Просмотр логов
	@echo "$(GREEN)Просмотр логов...$(NC)"
	docker-compose -f docker-compose.local.yml logs --tail=100

dashboard: ## Генерация дашбордов
	@echo "$(GREEN)Генерация дашбордов...$(NC)"
	$(PYTHON) src/ml_pipeline/monitoring/evidently_dashboard.py \
		--generate \
		--output $(REPORTS_DIR)/dashboards/evidently_dashboard.html
	@echo "$(GREEN)✓ Дашборды сгенерированы$(NC)"

# ============================================================================
## TERRAFORM (YANDEX CLOUD)
# ============================================================================

.PHONY: tf-init tf-plan tf-apply tf-destroy tf-output tf-refresh

tf-init: check-yc ## Инициализация Terraform
	@echo "$(GREEN)Инициализация Terraform...$(NC)"
	cd $(TF_DIR) && terraform init
	@echo "$(GREEN)✓ Terraform инициализирован$(NC)"

tf-plan: tf-init ## План развертывания инфраструктуры
	@echo "$(GREEN)Генерация плана Terraform...$(NC)"
	cd $(TF_DIR) && terraform plan -var-file=$(TF_VARS)
	@echo "$(GREEN)✓ План сгенерирован$(NC)"

tf-apply: tf-plan ## Развертывание инфраструктуры
	@echo "$(GREEN)Развертывание инфраструктуры в Yandex Cloud...$(NC)"
	cd $(TF_DIR) && terraform apply -var-file=$(TF_VARS) -auto-approve
	@echo "$(GREEN)✓ Инфраструктура развернута$(NC)"

tf-destroy: ## Уничтожение инфраструктуры
	@echo "$(RED)Уничтожение инфраструктуры в Yandex Cloud...$(NC)"
	cd $(TF_DIR) && terraform destroy -var-file=$(TF_VARS) -auto-approve
	@echo "$(GREEN)✓ Инфраструктура уничтожена$(NC)"

tf-output: ## Вывод выходных переменных Terraform
	@echo "$(GREEN)Выходные переменные Terraform:$(NC)"
	cd $(TF_DIR) && terraform output

tf-refresh: ## Обновление состояния Terraform
	@echo "$(GREEN)Обновление состояния Terraform...$(NC)"
	cd $(TF_DIR) && terraform refresh -var-file=$(TF_VARS)

# ============================================================================
# KUBERNETES (YANDEX CLOUD)
# ============================================================================

.PHONY: k8s-context k8s-deploy k8s-deploy-all k8s-status k8s-logs k8s-delete

k8s-context: check-k8s ## Настройка контекста Kubernetes
	@echo "$(GREEN)Настройка контекста Kubernetes...$(NC)"
	kubectl config use-context $(K8S_CONTEXT)
	@echo "$(GREEN)✓ Контекст настроен: $(K8S_CONTEXT)$(NC)"

k8s-deploy: k8s-context build-all ## Развертывание в Kubernetes
	@echo "$(GREEN)Развертывание в Kubernetes...$(NC)"
	kubectl apply -f $(K8S_DIR)/namespace.yaml
	kubectl apply -f $(K8S_DIR)/configs/ -n $(K8S_NAMESPACE)
	kubectl apply -f $(K8S_DIR)/secrets/ -n $(K8S_NAMESPACE)
	kubectl apply -f $(K8S_DIR)/deployments/ -n $(K8S_NAMESPACE)
	kubectl apply -f $(K8S_DIR)/services/ -n $(K8S_NAMESPACE)
	kubectl apply -f $(K8S_DIR)/ingress/ -n $(K8S_NAMESPACE)
	@echo "$(GREEN)✓ Приложение развернуто в Kubernetes$(NC)"

k8s-deploy-all: tf-apply k8s-deploy ## Полное развертывание в облаке

k8s-status: ## Статус Kubernetes ресурсов
	@echo "$(GREEN)Статус Kubernetes:$(NC)"
	@echo "$(CYAN)Поды:$(NC)"
	kubectl get pods -n $(K8S_NAMESPACE)
	@echo "$(CYAN)Сервисы:$(NC)"
	kubectl get services -n $(K8S_NAMESPACE)
	@echo "$(CYAN)Deployments:$(NC)"
	kubectl get deployments -n $(K8S_NAMESPACE)
	@echo "$(CYAN)Ingress:$(NC)"
	kubectl get ingress -n $(K8S_NAMESPACE)

k8s-logs: ## Просмотр логов Kubernetes
	@echo "$(GREEN)Логи Kubernetes:$(NC)"
	kubectl logs -n $(K8S_NAMESPACE) --selector=app=credit-scoring-api --tail=100

k8s-delete: ## Удаление развертывания из Kubernetes
	@echo "$(YELLOW)Удаление развертывания из Kubernetes...$(NC)"
	kubectl delete -f $(K8S_DIR)/ --ignore-not-found=true -n $(K8S_NAMESPACE)
	kubectl delete namespace $(K8S_NAMESPACE) --ignore-not-found=true
	@echo "$(GREEN)✓ Развертывание удалено$(NC)"

# ============================================================================
# CI/CD И АВТОМАТИЗАЦИЯ
# ============================================================================

.PHONY: ci-build ci-test ci-deploy-staging ci-deploy-production

ci-build: setup test build-all ## CI: Сборка и тестирование
	@echo "$(GREEN)✓ CI: Сборка и тестирование завершены$(NC)"

ci-test: test-unit test-integration test-api ## CI: Запуск всех тестов
	@echo "$(GREEN)✓ CI: Все тесты пройдены$(NC)"

ci-deploy-staging: ## CI: Деплой в staging
	@echo "$(GREEN)CI: Деплой в staging...$(NC)"
	ENVIRONMENT=staging $(MAKE) tf-apply
	ENVIRONMENT=staging $(MAKE) k8s-deploy
	@echo "$(GREEN)✓ CI: Деплой в staging завершен$(NC)"

ci-deploy-production: ## CI: Деплой в production
	@echo "$(GREEN)CI: Деплой в production...$(NC)"
	ENVIRONMENT=production $(MAKE) tf-apply
	ENVIRONMENT=production $(MAKE) k8s-deploy
	@echo "$(GREEN)✓ CI: Деплой в production завершен$(NC)"

# ============================================================================
# БЕЗОПАСНОСТЬ И СКАНИРОВАНИЕ
# ============================================================================

.PHONY: security-scan vulnerability-scan secrets-check

security-scan: ## Сканирование безопасности
	@echo "$(GREEN)Сканирование безопасности...$(NC)"
	bandit -r src/ -f html -o $(REPORTS_DIR)/security_scan.html
	@echo "$(GREEN)✓ Сканирование безопасности завершено$(NC)"

vulnerability-scan: ## Сканирование уязвимостей Docker образов
	@echo "$(GREEN)Сканирование уязвимостей Docker образов...$(NC)"
	trivy image $(IMAGE_API)
	trivy image $(IMAGE_TRAINING)
	@echo "$(GREEN)✓ Сканирование уязвимостей завершено$(NC)"

secrets-check: ## Проверка на утечку секретов
	@echo "$(GREEN)Проверка на утечку секретов...$(NC)"
	gitleaks detect --source . --verbose
	@echo "$(GREEN)✓ Проверка секретов завершена$(NC)"

# ============================================================================
## ОЧИСТКА
# ============================================================================

.PHONY: clean clean-models clean-data clean-docker clean-terraform clean-all

clean: ## Очистка временных файлов
	@echo "$(GREEN)Очистка временных файлов...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	@echo "$(GREEN)✓ Временные файлы очищены$(NC)"

clean-models: ## Очистка моделей
	@echo "$(GREEN)Очистка моделей...$(NC)"
	rm -rf $(MODELS_DIR)/*.pth
	rm -rf $(MODELS_DIR)/*.onnx
	rm -rf $(MODELS_DIR)/*.joblib
	rm -rf $(MODELS_DIR)/deployed/*
	@echo "$(GREEN)✓ Модели очищены$(NC)"

clean-data: ## Очистка данных
	@echo "$(GREEN)Очистка данных...$(NC)"
	rm -rf $(DATA_DIR)/processed/*
	@echo "$(GREEN)✓ Данные очищены$(NC)"

clean-docker: ## Очистка Docker
	@echo "$(GREEN)Очистка Docker...$(NC)"
	docker system prune -f
	docker volume prune -f
	@echo "$(GREEN)✓ Docker очищен$(NC)"

clean-terraform: ## Очистка Terraform
	@echo "$(GREEN)Очистка Terraform...$(NC)"
	rm -rf $(TF_DIR)/.terraform
	rm -f $(TF_DIR)/$(TF_STATE)*
	rm -f $(TF_DIR)/*.tfstate.backup
	@echo "$(GREEN)✓ Terraform очищен$(NC)"

clean-all: clean clean-models clean-data clean-docker clean-terraform local-down ## Полная очистка
	@echo "$(GREEN)✓ Полная очистка завершена$(NC)"

# ============================================================================
## УТИЛИТЫ И ДЕМОНСТРАЦИЯ
# ============================================================================

.PHONY: demo demo-api demo-ml demo-monitoring demo-all

demo-api: ## Демонстрация API
	@echo "$(GREEN)Демонстрация API...$(NC)"
	@echo "1. Откройте в браузере: http://localhost:8000"
	@echo "2. Откройте Swagger UI: http://localhost:8000/api/docs"
	@echo "3. Протестируйте эндпоинты:"
	@echo "   - GET /health"
	@echo "   - POST /api/v1/predict"
	@echo "   - GET /api/v1/services"

demo-ml: ## Демонстрация ML компонентов
	@echo "$(GREEN)Демонстрация ML компонентов...$(NC)"
	@echo "1. MLflow: http://localhost:5000"
	@echo "2. Обучение модели: make train"
	@echo "3. Конвертация в ONNX: make convert"
	@echo "4. Оптимизация: make optimize"

demo-monitoring: ## Демонстрация мониторинга
	@echo "$(GREEN)Демонстрация мониторинга...$(NC)"
	@echo "1. Grafana: http://localhost:3000 (admin/admin)"
	@echo "2. Prometheus: http://localhost:9090"
	@echo "3. Loki: http://localhost:3100"
	@echo "4. Мониторинг дрифта: make monitor-drift"

demo-all: demo-api demo-ml demo-monitoring ## Вся демонстрация

# ============================================================================
## ДОКУМЕНТАЦИЯ
# ============================================================================

.PHONY: docs docs-api docs-ml docs-infra

docs: ## Генерация всей документации
	@echo "$(GREEN)Генерация документации...$(NC)"
	pdoc --html src --output-dir docs/api --force
	@echo "$(GREEN)✓ Документация сгенерирована$(NC)"

docs-api: ## Документация API
	@echo "$(GREEN)Генерация документации API...$(NC)"
	redoc-cli bundle openapi.yaml -o docs/api.html
	@echo "$(GREEN)✓ Документация API сгенерирована$(NC)"

docs-ml: ## Документация ML пайплайна
	@echo "$(GREEN)Генерация ML документации...$(NC)"
	$(PYTHON) scripts/generate_ml_docs.py --output docs/ml/
	@echo "$(GREEN)✓ ML документация сгенерирована$(NC)"

docs-infra: ## Документация инфраструктуры
	@echo "$(GREEN)Генерация инфраструктурной документации...$(NC)"
	terraform-docs markdown $(TF_DIR) > $(TF_DIR)/README.md
	@echo "$(GREEN)✓ Инфраструктурная документация сгенерирована$(NC)"

# ============================================
# РАБОЧИЕ ЦЕЛИ ДЛЯ ЗАПУСКА ВСЕГО
# ============================================

create-missing:
	@echo "Создание недостающих файлов..."
	python scripts/create_missing_files.py

health-check:
	@echo "Проверка здоровья сервисов..."
ifeq ($(SYSTEM),Windows)
	powershell -ExecutionPolicy Bypass -File scripts/windows/health-check.ps1
else
	chmod +x scripts/linux/health-check.sh
	./scripts/linux/health-check.sh
endif

run-local: create-missing
	@echo "Запуск всех сервисов локально..."
	docker-compose down
	docker-compose build
	docker-compose up -d
	@echo "Ожидание запуска сервисов..."

	@echo "Проверка доступности..."
	curl -f http://localhost:8000/health || echo "API пока не готов"
	curl -f http://localhost:5000 || echo "MLflow пока не готов"
	@echo ""
	@echo "Сервисы запущены:"
	@echo "  API:      http://localhost:8000"
	@echo "  MLflow:   http://localhost:5000"
	@echo "  MinIO:    http://localhost:9001"
	@echo "  Grafana:  http://localhost:3000"
	@echo ""
	@echo "Для просмотра логов: docker-compose logs -f"
	@echo "Для остановки:       docker-compose down"

restart-api:
	@echo "Перезапуск API..."
	docker-compose stop api
	docker-compose build api
	docker-compose up -d api
	@echo "Проверка API..."
	curl -f -s http://localhost:8000/health && echo "API работает" || echo "API не отвечает, проверьте логи: docker-compose logs api"

status:
	@echo "Статус контейнеров:"
	docker-compose ps
	@echo ""
	@echo "Использование ресурсов:"
	docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" | head -n 8

clean-docker:
	@echo "Очистка Docker..."
	docker-compose down -v
	docker system prune -f
	@echo "Docker очищен"

# Полный рабочий пайплайн
full-pipeline: create-missing run-local health-check
	@echo "Все сервисы запущены и проверены!"

# ============================================================================
## КОНЕЦ ФАЙЛА
# ============================================================================

# Автоматическая документация целей
.DEFAULT_GOAL := help

