#!/bin/bash
set -e

# Blue-green deployment script
# Этап 4: Стратегии развертывания

BLUE_APP="credit-scoring-api-blue"
GREEN_APP="credit-scoring-api-green"
NAMESPACE="${K8S_NAMESPACE:-credit-scoring}"
IMAGE="${IMAGE_REGISTRY:-cr.yandex.cloud/credit-scoring/credit-scoring-api}"
TAG="${IMAGE_TAG:-latest}"
INGRESS_NAME="credit-scoring-api"
SERVICE_NAME="credit-scoring-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Проверка предварительных условий..."
    
    # Проверка kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl не установлен"
        exit 1
    fi
    
    # Проверка подключения к кластеру
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Не удалось подключиться к Kubernetes кластеру"
        exit 1
    fi
    
    # Проверка namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warn "Namespace $NAMESPACE не существует, создаем..."
        kubectl create namespace "$NAMESPACE"
    fi
}

get_current_version() {
    # Определяем текущую активную версию (blue или green)
    local service_selector=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.app}' 2>/dev/null || true)
    
    if [[ "$service_selector" == "$BLUE_APP" ]]; then
        echo "blue"
    elif [[ "$service_selector" == "$GREEN_APP" ]]; then
        echo "green"
    else
        echo "unknown"
    fi
}

deploy_new_version() {
    local current_version=$1
    local new_version=$2
    
    log_info "Развертывание новой версии: $new_version"
    
    # Определяем deployment для новой версии
    if [[ "$new_version" == "green" ]]; then
        DEPLOYMENT_NAME="$GREEN_APP"
        OLD_DEPLOYMENT_NAME="$BLUE_APP"
    else
        DEPLOYMENT_NAME="$BLUE_APP"
        OLD_DEPLOYMENT_NAME="$GREEN_APP"
    fi
    
    # Создаем deployment манифест
    cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $DEPLOYMENT_NAME
  labels:
    app: $DEPLOYMENT_NAME
    version: $TAG
spec:
  replicas: 2
  selector:
    matchLabels:
      app: $DEPLOYMENT_NAME
  template:
    metadata:
      labels:
        app: $DEPLOYMENT_NAME
        version: $TAG
    spec:
      containers:
      - name: api
        image: $IMAGE:$TAG
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DEPLOYMENT_COLOR
          value: "$new_version"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
EOF
    
    # Ждем готовности нового deployment
    log_info "Ожидание готовности новой версии..."
    kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=300s
    
    # Создаем service для новой версии (для тестирования)
    cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Service
metadata:
  name: $DEPLOYMENT_NAME
spec:
  selector:
    app: $DEPLOYMENT_NAME
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
EOF
    
    log_info "Новая версия развернута и готова для тестирования"
}

test_new_version() {
    local new_version=$1
    
    log_info "Тестирование новой версии..."
    
    if [[ "$new_version" == "green" ]]; then
        SERVICE_NAME_TEST="$GREEN_APP"
    else
        SERVICE_NAME_TEST="$BLUE_APP"
    fi
    
    # Получаем IP сервиса для тестирования
    local service_ip=$(kubectl get service "$SERVICE_NAME_TEST" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    if [[ -z "$service_ip" ]]; then
        log_error "Не удалось получить IP сервиса для тестирования"
        return 1
    fi
    
    # Тестируем health endpoint
    log_info "Проверка здоровья новой версии..."
    for i in {1..10}; do
        if kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl:8.2.1 \
            -- curl -f http://"$service_ip":80/health/live --max-time 5; then
            log_info "Health check пройден"
            break
        else
            if [[ $i -eq 10 ]]; then
                log_error "Health check не пройден после 10 попыток"
                return 1
            fi
            sleep 5
        fi
    done
    
    # Тестируем основной эндпоинт
    log_info "Тестирование эндпоинта предсказаний..."
    kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl:8.2.1 \
        -- curl -X POST http://"$service_ip":80/api/v1/predict \
        -H "Content-Type: application/json" \
        -d '{"features": {"age": 30, "income": 50000, "credit_amount": 10000, "loan_duration": 12, "payment_to_income": 0.2}}' \
        --max-time 10
    
    log_info "Тестирование завершено успешно"
    return 0
}

switch_traffic() {
    local new_version=$1
    
    log_info "Перенаправление трафика на новую версию: $new_version"
    
    # Обновляем основной service чтобы он указывал на новую версию
    if [[ "$new_version" == "green" ]]; then
        NEW_SELECTOR="$GREEN_APP"
    else
        NEW_SELECTOR="$BLUE_APP"
    fi
    
    kubectl patch service "$SERVICE_NAME" -n "$NAMESPACE" -p '{"spec":{"selector":{"app":"'"$NEW_SELECTOR"'"}}}'
    
    # Ждем пока обновится service
    sleep 10
    
    log_info "Трафик перенаправлен на $new_version версию"
}

cleanup_old_version() {
    local old_version=$1
    
    log_info "Очистка старой версии: $old_version"
    
    if [[ "$old_version" == "green" ]]; then
        OLD_DEPLOYMENT="$GREEN_APP"
        OLD_SERVICE="$GREEN_APP"
    else
        OLD_DEPLOYMENT="$BLUE_APP"
        OLD_SERVICE="$BLUE_APP"
    fi
    
    # Удаляем старый deployment и service
    kubectl delete deployment "$OLD_DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete service "$OLD_SERVICE" -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "Старая версия очищена"
}

main() {
    log_info "Начало blue-green деплоя"
    log_info "Image: $IMAGE:$TAG"
    log_info "Namespace: $NAMESPACE"
    
    check_prerequisites
    
    # Определяем текущую и новую версии
    CURRENT_VERSION=$(get_current_version)
    
    if [[ "$CURRENT_VERSION" == "blue" ]]; then
        NEW_VERSION="green"
    elif [[ "$CURRENT_VERSION" == "green" ]]; then
        NEW_VERSION="blue"
    else
        # Первый деплой, используем blue
        CURRENT_VERSION="none"
        NEW_VERSION="blue"
    fi
    
    log_info "Текущая версия: $CURRENT_VERSION"
    log_info "Новая версия: $NEW_VERSION"
    
    # Развертываем новую версию
    deploy_new_version "$CURRENT_VERSION" "$NEW_VERSION"
    
    # Тестируем новую версию
    if ! test_new_version "$NEW_VERSION"; then
        log_error "Тестирование новой версии не пройдено. Отмена деплоя."
        exit 1
    fi
    
    # Перенаправляем трафик
    switch_traffic "$NEW_VERSION"
    
    # Очищаем старую версию
    if [[ "$CURRENT_VERSION" != "none" ]]; then
        cleanup_old_version "$CURRENT_VERSION"
    fi
    
    # Финальная проверка
    log_info "Проверка финального состояния..."
    kubectl get deployments -n "$NAMESPACE" -l "app in ($BLUE_APP, $GREEN_APP)"
    kubectl get services -n "$NAMESPACE" -l "app in ($BLUE_APP, $GREEN_APP)"
    
    log_info "Blue-green деплой завершен успешно!"
}

# Обработка аргументов командной строки
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --namespace)
            K8S_NAMESPACE="$2"
            shift 2
            ;;
        --registry)
            IMAGE_REGISTRY="$2"
            shift 2
            ;;
        --help)
            echo "Использование: $0 [--image-tag TAG] [--namespace NAMESPACE] [--registry REGISTRY]"
            echo ""
            echo "Примеры:"
            echo "  $0 --image-tag v1.2.3 --namespace production"
            echo "  $0 --image-tag latest --namespace staging"
            exit 0
            ;;
        *)
            log_error "Неизвестный аргумент: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

main