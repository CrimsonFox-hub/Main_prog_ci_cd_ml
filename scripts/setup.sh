#!/bin/bash
set -e

# Script: setup.sh
# Настройка и развертывание MLOps проекта для кредитного скоринга

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функции для вывода
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Переменные окружения
PROJECT_NAME="credit-scoring"
ENVIRONMENT="${ENVIRONMENT:-staging}"
REGISTRY="${REGISTRY:-cr.yandex.cloud}"
IMAGE_NAME="${REGISTRY}/${YC_REGISTRY_ID:-credit-scoring}"
K8S_NAMESPACE="credit-scoring"
TERRAFORM_DIR="infrastructure/environments/${ENVIRONMENT}"

check_dependencies() {
    log_info "Проверка зависимостей..."
    
    local missing_deps=0
    
    # Проверка Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker не установлен"
        missing_deps=1
    fi
    
    # Проверка Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose не установлен"
        missing_deps=1
    fi
    
    # Проверка Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 не установлен"
        missing_deps=1
    fi
    
    # Проверка pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 не установлен"
        missing_deps=1
    fi
    
    if [ $missing_deps -eq 0 ]; then
        log_info "Все зависимости установлены"
    else
        log_error "Установите недостающие зависимости"
        exit 1
    fi
}

check_cloud_dependencies() {
    log_info "Проверка облачных зависимостей..."
    
    local missing_cloud_deps=0
    
    # Проверка Yandex Cloud CLI (опционально)
    if ! command -v yc &> /dev/null; then
        log_warn "YC CLI не установлен (опционально для локальной разработки)"
    fi
    
    # Проверка kubectl
    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl не установлен (опционально для локальной разработки)"
    fi
    
    # Проверка Terraform
    if ! command -v terraform &> /dev/null; then
        log_warn "Terraform не установлен (опционально для локальной разработки)"
    fi
    
    if [ $missing_cloud_deps -eq 0 ]; then
        log_info "Облачные зависимости проверены"
    fi
}

setup_python_environment() {
    log_info "Настройка Python окружения..."
    
    # Создание виртуального окружения
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_info "Виртуальное окружение создано"
    fi
    
    # Активация виртуального окружения
    source venv/bin/activate
    
    # Установка зависимостей
    if [ -f "requirements.txt" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
        log_info "Основные зависимости установлены"
    fi
    
    if [ -f "requirements-ml.txt" ]; then
        pip install -r requirements-ml.txt
        log_info "ML зависимости установлены"
    fi
    
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        log_info "Зависимости для разработки установлены"
    fi
}

setup_configuration() {
    log_info "Настройка конфигурации..."
    
    # Создание директорий
    mkdir -p data/{raw,processed}
    mkdir -p models/{trained,processed,backup,retrained}
    mkdir -p configs/{local,staging,production}
    mkdir -p reports/{benchmarks,drift_monitoring,load_tests}
    mkdir -p logs/{api,training,monitoring}
    mkdir -p kubernetes/{base,overlays/{staging,production}}
    
    # Копирование примеров конфигурации
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_warn "Создан файл .env из примера. Отредактируйте его перед использованием."
        else
            log_warn "Создайте файл .env с настройками окружения"
        fi
    fi
    
    # Создание конфигурационных файлов если их нет
    if [ ! -f "configs/local/api.yaml" ]; then
        cat > configs/local/api.yaml << EOF
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: true

logging:
  level: "DEBUG"
  format: "json"

model:
  type: "pkl"
  path: "models/trained/credit_scoring_model.pkl"
  cache_enabled: false

database:
  url: "postgresql://postgres:postgres@localhost:5432/credit_scoring"

redis:
  url: "redis://localhost:6379/0"
EOF
        log_info "Создан конфиг для локальной разработки"
    fi
}

setup_data() {
    log_info "Настройка данных..."
    
    # Проверка существования данных
    if [ ! -f "data/processed/train.csv" ] || [ ! -f "data/processed/test.csv" ]; then
        log_info "Создание тестовых данных..."
        
        # Используем скрипт для создания данных
        if [ -f "scripts/data/create_sample_data.py" ]; then
            python scripts/data/create_sample_data.py
        elif [ -f "scripts/data/download_data.py" ]; then
            python scripts/data/download_data.py --use-sample
        else
            log_warn "Скрипты для данных не найдены, создаем простые данные..."
            
            # Создаем минимальный набор данных
            mkdir -p data/processed
            cat > data/processed/train.csv << EOF
age,income,credit_amount,loan_duration,payment_to_income,existing_credits,dependents,residence_since,installment_rate,default
30,50000,10000,12,0.2,1,2,5,2.5,0
45,80000,20000,24,0.25,2,1,10,3.0,0
25,30000,5000,6,0.17,0,0,2,1.8,1
35,60000,15000,18,0.25,1,3,7,2.7,0
EOF
            
            cat > data/processed/test.csv << EOF
age,income,credit_amount,loan_duration,payment_to_income,existing_credits,dependents,residence_since,installment_rate,default
28,45000,8000,12,0.18,0,1,3,2.2,0
50,100000,30000,36,0.3,3,2,15,3.5,1
EOF
            
            log_info "Созданы простые тестовые данные"
        fi
    else
        log_info "Данные уже существуют"
    fi
}

setup_docker() {
    log_info "Настройка Docker..."
    
    # Сборка образов для локальной разработки
    if [ "$ENVIRONMENT" = "local" ]; then
        log_info "Сборка Docker образов для локальной разработки..."
        
        # Проверка существования Dockerfile
        if [ -f "docker/api/Dockerfile" ]; then
            docker build -f docker/api/Dockerfile -t ${IMAGE_NAME}-api:local .
        else
            log_warn "Dockerfile для API не найден"
        fi
        
        if [ -f "docker/training/Dockerfile" ]; then
            docker build -f docker/training/Dockerfile -t ${IMAGE_NAME}-training:local .
        fi
        
        # Запуск docker-compose
        if [ -f "docker-compose.yml" ]; then
            log_info "Запуск docker-compose..."
            docker-compose up -d
        fi
    fi
}

setup_terraform() {
    if [ "$ENVIRONMENT" != "local" ] && [ -d "$TERRAFORM_DIR" ]; then
        log_info "Настройка Terraform для окружения: $ENVIRONMENT"
        
        # Инициализация Terraform
        cd $TERRAFORM_DIR
        
        if [ -f "backend.tf" ]; then
            # Используем remote state
            terraform init -backend-config="access_key=$YC_ACCESS_KEY" \
                          -backend-config="secret_key=$YC_SECRET_KEY"
        else
            terraform init
        fi
        
        # Создание terraform.tfvars если не существует
        if [ ! -f "terraform.tfvars" ] && [ -f "terraform.tfvars.example" ]; then
            cp terraform.tfvars.example terraform.tfvars
            log_warn "Создан terraform.tfvars из примера. Отредактируйте его перед использованием."
        fi
        
        cd - > /dev/null
    fi
}

setup_kubernetes() {
    if [ "$ENVIRONMENT" != "local" ] && command -v kubectl &> /dev/null; then
        log_info "Настройка Kubernetes..."
        
        # Проверка подключения к кластеру
        if kubectl cluster-info &> /dev/null; then
            # Создание namespace если не существует
            if ! kubectl get namespace $K8S_NAMESPACE &> /dev/null; then
                kubectl create namespace $K8S_NAMESPACE
                log_info "Создан namespace: $K8S_NAMESPACE"
            fi
            
            # Применение базовых конфигураций
            if [ -d "kubernetes/base" ]; then
                kubectl apply -f kubernetes/base/ -n $K8S_NAMESPACE
            fi
        else
            log_warn "Не удалось подключиться к Kubernetes кластеру"
        fi
    fi
}

run_tests() {
    log_info "Запуск тестов..."
    
    # Проверка Python тестов
    if [ -d "tests" ]; then
        python -m pytest tests/unit/ -v --tb=short
    else
        log_warn "Директория tests не найдена"
    fi
    
    # Проверка качества кода
    if command -v black &> /dev/null; then
        black --check src/ scripts/ || log_warn "Форматирование кода не соответствует black"
    fi
    
    if command -v flake8 &> /dev/null; then
        flake8 src/ scripts/ --max-line-length=100 || log_warn "Найдены проблемы с стилем кода"
    fi
}

show_summary() {
    log_info "=" * 60
    log_info "НАСТРОЙКА ЗАВЕРШЕНА"
    log_info "=" * 60
    
    echo ""
    echo "Проект: $PROJECT_NAME"
    echo "Окружение: $ENVIRONMENT"
    echo ""
    
    if [ "$ENVIRONMENT" = "local" ]; then
        echo "Локальные сервисы:"
        echo "  - API: http://localhost:8000"
        echo "  - API Docs: http://localhost:8000/docs"
        echo "  - MLflow: http://localhost:5000"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3000"
        echo ""
        echo "Управление:"
        echo "  docker-compose up -d     # Запуск сервисов"
        echo "  docker-compose down      # Остановка сервисов"
        echo "  docker-compose logs -f   # Просмотр логов"
    else
        echo "Облачная инфраструктура:"
        echo "  - Terraform директория: $TERRAFORM_DIR"
        echo "  - Kubernetes namespace: $K8S_NAMESPACE"
        echo "  - Container registry: $REGISTRY"
        echo ""
        echo "Следующие шаги:"
        echo "  1. Отредактируйте .env и terraform.tfvars"
        echo "  2. Запустите terraform apply для создания инфраструктуры"
        echo "  3. Соберите и загрузите Docker образы"
        echo "  4. Разверните приложение в Kubernetes"
    fi
    
    echo ""
    echo "Полезные команды:"
    echo "  make help                    # Список всех команд"
    echo "  python scripts/setup.py      # Дополнительная настройка"
    echo "  ./scripts/train_model.py     # Обучение модели"
    echo "  ./scripts/run_tests.sh       # Запуск тестов"
}

main() {
    log_info "Начало настройки MLOps проекта для кредитного скоринга"
    log_info "Окружение: $ENVIRONMENT"
    
    # Проверка зависимостей
    check_dependencies
    check_cloud_dependencies
    
    # Настройка окружения
    setup_python_environment
    setup_configuration
    setup_data
    
    # Настройка инфраструктуры
    if [ "$ENVIRONMENT" != "local" ]; then
        setup_terraform
        setup_kubernetes
    else
        setup_docker
    fi
    
    # Запуск тестов
    run_tests
    
    # Итоговая информация
    show_summary
    
    log_info "Настройка завершена успешно!"
}

# Обработка аргументов командной строки
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment|-e)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Использование: $0 [--environment ENV]"
            echo "Доступные окружения: local, staging, production"
            echo ""
            echo "Примеры:"
            echo "  $0 --environment local      # Локальная разработка"
            echo "  $0 --environment staging    # Staging окружение"
            exit 0
            ;;
        *)
            log_error "Неизвестный аргумент: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# Запуск основной функции
main