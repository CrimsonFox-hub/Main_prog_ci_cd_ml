#!/bin/bash
set -e

# Canary deployment script

APP_NAME="credit-scoring-api"
NAMESPACE="${K8S_NAMESPACE:-credit-scoring}"
IMAGE="${IMAGE_REGISTRY:-cr.yandex.cloud/credit-scoring/credit-scoring-api}"
TAG="${IMAGE_TAG:-latest}"
CANARY_PERCENTAGE="${CANARY_PERCENTAGE:-10}"
CANARY_DURATION="${CANARY_DURATION:-300}"
MAX_PERCENTAGE="${MAX_PERCENTAGE:-50}"

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

deploy_canary() {
    log_info "Запуск canary-развертывания..."
    log_info "Image: $IMAGE:$TAG"
    log_info "Canary процент: $CANARY_PERCENTAGE%"
    log_info "Длительность canary: $CANARY_DURATION секунд"
    
    # Создаем canary deployment
    CANARY_DEPLOYMENT_NAME="$APP_NAME-canary"
    
    log_info "Создание canary deployment: $CANARY_DEPLOYMENT_NAME"
    
    cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $CANARY_DEPLOYMENT_NAME
  labels:
    app: $APP_NAME
    version: canary-$TAG
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $APP_NAME
      version: canary-$TAG
  template:
    metadata:
      labels:
        app: $APP_NAME
        version: canary-$TAG
    spec:
      containers:
      - name: api
        image: $IMAGE:$TAG
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "canary"
        - name: DEPLOYMENT_VERSION
          value: "canary-$TAG"
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
    
    # Ждем готовности canary deployment
    log_info "Ожидание готовности canary deployment..."
    kubectl rollout status deployment/"$CANARY_DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=300s
    
    # Создаем service для canary, если еще нет
    if ! kubectl get service "$APP_NAME-canary" -n "$NAMESPACE" &> /dev/null; then
        log_info "Создание service для canary..."
        cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Service
metadata:
  name: $APP_NAME-canary
spec:
  selector:
    app: $APP_NAME
    version: canary-$TAG
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
EOF
    fi
    
    log_info "Canary deployment создан и готов"
}

configure_istio_traffic() {
    log_info "Настройка распределения трафика с помощью Istio..."
    
    # Проверяем, установлен ли Istio
    if ! kubectl get namespace istio-system &> /dev/null; then
        log_error "Istio не установлен в кластере"
        return 1
    fi
    
    # Создаем VirtualService для canary routing
    log_info "Создание VirtualService для маршрутизации трафика..."
    
    cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: $APP_NAME
spec:
  hosts:
  - $APP_NAME.$NAMESPACE.svc.cluster.local
  http:
  - route:
    - destination:
        host: $APP_NAME.$NAMESPACE.svc.cluster.local
      weight: $((100 - CANARY_PERCENTAGE))
    - destination:
        host: $APP_NAME-canary.$NAMESPACE.svc.cluster.local
      weight: $CANARY_PERCENTAGE
EOF
    
    log_info "Трафик настроен: $((100 - CANARY_PERCENTAGE))% -> основная версия, $CANARY_PERCENTAGE% -> canary"
}

monitor_canary() {
    local duration=$1
    local interval=10
    local iterations=$((duration / interval))
    
    log_info "Мониторинг canary в течение $duration секунд (интервал: $interval сек)..."
    
    for ((i=1; i<=iterations; i++)); do
        log_info "Проверка #$i из $iterations"
        
        # Проверяем метрики canary
        check_canary_metrics
        
        # Проверяем ошибки в логах canary
        check_canary_logs
        
        # Если есть критические ошибки, прерываем canary
        if check_critical_errors; then
            log_error "Обнаружены критические ошибки в canary. Прерывание..."
            return 1
        fi
        
        # Если это не последняя итерация, ждем
        if [[ $i -lt $iterations ]]; then
            sleep $interval
        fi
    done
    
    log_info "Мониторинг canary завершен"
    return 0
}

check_canary_metrics() {
    # Здесь можно добавить проверку метрик, например, через Prometheus
    # Для простоты проверим доступность и время ответа
    
    local canary_service="$APP_NAME-canary.$NAMESPACE.svc.cluster.local"
    
    log_info "Проверка метрик canary..."
    
    # Используем временный pod для проверки доступности
    if kubectl run -it --rm --restart=Never --image=curlimages/curl:8.2.1 test-canary -- \
        curl -f http://$canary_service/health/live --max-time 5 &> /dev/null; then
        log_info "Canary health check: OK"
    else
        log_warn "Canary health check: FAILED"
    fi
}

check_canary_logs() {
    log_info "Проверка логов canary на ошибки..."
    
    # Получаем последние 10 строк логов canary deployment
    local logs=$(kubectl logs -n "$NAMESPACE" -l "app=$APP_NAME,version=canary-$TAG" --tail=10 2>/dev/null || true)
    
    # Ищем ошибки в логах (простой grep)
    if echo "$logs" | grep -i "error\|exception\|fail" | head -5; then
        log_warn "Найдены ошибки в логах canary"
    else
        log_info "В логах canary ошибок не обнаружено"
    fi
}

check_critical_errors() {
    # Здесь можно реализовать более сложную логику проверки критических ошибок
    # Например, проверка метрик через Prometheus API
    
    # Временно возвращаем false (нет критических ошибок)
    return 1
}

increase_traffic() {
    local target_percentage=$1
    
    log_info "Увеличение трафика на canary до $target_percentage%"
    
    if ! kubectl get virtualservice "$APP_NAME" -n "$NAMESPACE" &> /dev/null; then
        log_error "VirtualService не найден"
        return 1
    fi
    
    # Обновляем VirtualService с новым распределением трафика
    local main_weight=$((100 - target_percentage))
    
    kubectl patch virtualservice "$APP_NAME" -n "$NAMESPACE" --type=merge -p "
spec:
  http:
  - route:
    - destination:
        host: $APP_NAME.$NAMESPACE.svc.cluster.local
      weight: $main_weight
    - destination:
        host: $APP_NAME-canary.$NAMESPACE.svc.cluster.local
      weight: $target_percentage
"
    
    log_info "Трафик обновлен: $main_weight% -> основная версия, $target_percentage% -> canary"
}

promote_canary() {
    log_info "Продвижение canary до основной версии..."
    
    # Масштабируем canary deployment до полного размера
    local current_replicas=$(kubectl get deployment "$APP_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    
    log_info "Масштабирование canary до $current_replicas реплик..."
    kubectl scale deployment "$APP_NAME-canary" -n "$NAMESPACE" --replicas="$current_replicas"
    
    # Ждем масштабирования
    kubectl rollout status deployment/"$APP_NAME-canary" -n "$NAMESPACE" --timeout=300s
    
    # Перенаправляем весь трафик на canary
    log_info "Перенаправление всего трафика на canary..."
    kubectl patch virtualservice "$APP_NAME" -n "$NAMESPACE" --type=merge -p "
spec:
  http:
  - route:
    - destination:
        host: $APP_NAME-canary.$NAMESPACE.svc.cluster.local
      weight: 100
"
    
    sleep 10
    
    # Обновляем основной deployment до новой версии
    log_info "Обновление основного deployment..."
    kubectl set image deployment/"$APP_NAME" -n "$NAMESPACE" "api=$IMAGE:$TAG"
    kubectl rollout status deployment/"$APP_NAME" -n "$NAMESPACE" --timeout=300s
    
    # Возвращаем трафик на основной deployment
    log_info "Возврат трафика на основной deployment..."
    kubectl patch virtualservice "$APP_NAME" -n "$NAMESPACE" --type=merge -p "
spec:
  http:
  - route:
    - destination:
        host: $APP_NAME.$NAMESPACE.svc.cluster.local
      weight: 100
"
    
    # Удаляем canary deployment
    log_info "Удаление canary deployment..."
    kubectl delete deployment "$APP_NAME-canary" -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete service "$APP_NAME-canary" -n "$NAMESPACE" --ignore-not-found=true
    
    # Восстанавливаем VirtualService
    kubectl delete virtualservice "$APP_NAME" -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "Canary успешно продвинут до основной версии!"
}

rollback_canary() {
    log_info "Откат canary..."
    
    # Возвращаем весь трафик на основной deployment
    if kubectl get virtualservice "$APP_NAME" -n "$NAMESPACE" &> /dev/null; then
        kubectl patch virtualservice "$APP_NAME" -n "$NAMESPACE" --type=merge -p "
spec:
  http:
  - route:
    - destination:
        host: $APP_NAME.$NAMESPACE.svc.cluster.local
      weight: 100
"
    fi
    
    # Удаляем canary deployment
    kubectl delete deployment "$APP_NAME-canary" -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete service "$APP_NAME-canary" -n "$NAMESPACE" --ignore-not-found=true
    
    # Удаляем VirtualService если создавали
    kubectl delete virtualservice "$APP_NAME" -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "Canary откатан. Весь трафик возвращен на основную версию."
}

main() {
    log_info "Начало canary-развертывания"
    
    check_prerequisites
    
    # Шаг 1: Развертывание canary
    deploy_canary
    
    # Шаг 2: Настройка маршрутизации трафика (если есть Istio)
    if [[ "$USE_ISTIO" == "true" ]]; then
        if ! configure_istio_traffic; then
            log_warn "Не удалось настроить Istio, используем простую стратегию"
            USE_ISTIO="false"
        fi
    fi
    
    # Шаг 3: Мониторинг canary с начальным процентом трафика
    if ! monitor_canary "$CANARY_DURATION"; then
        log_error "Обнаружены проблемы в canary. Откат..."
        rollback_canary
        exit 1
    fi
    
    # Шаг 4: Постепенное увеличение трафика (если процент меньше максимального)
    if [[ "$CANARY_PERCENTAGE" -lt "$MAX_PERCENTAGE" ]]; then
        local next_percentage=$((CANARY_PERCENTAGE * 2))
        
        while [[ "$next_percentage" -le "$MAX_PERCENTAGE" ]]; do
            log_info "Увеличение трафика на canary до $next_percentage%"
            
            if [[ "$USE_ISTIO" == "true" ]]; then
                increase_traffic "$next_percentage"
            fi
            
            if ! monitor_canary "$CANARY_DURATION"; then
                log_error "Проблемы при $next_percentage% трафика. Откат..."
                rollback_canary
                exit 1
            fi
            
            next_percentage=$((next_percentage * 2))
        done
    fi
    
    # Шаг 5: Продвижение canary до основной версии
    promote_canary
    
    log_info "Canary-развертывание успешно завершено!"
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
        --canary-percentage)
            CANARY_PERCENTAGE="$2"
            shift 2
            ;;
        --canary-duration)
            CANARY_DURATION="$2"
            shift 2
            ;;
        --max-percentage)
            MAX_PERCENTAGE="$2"
            shift 2
            ;;
        --use-istio)
            USE_ISTIO="$2"
            shift 2
            ;;
        --help)
            echo "Использование: $0 [опции]"
            echo ""
            echo "Опции:"
            echo "  --image-tag TAG           Тег образа (по умолчанию: latest)"
            echo "  --namespace NAMESPACE     Kubernetes namespace (по умолчанию: credit-scoring)"
            echo "  --registry REGISTRY       Registry образа"
            echo "  --canary-percentage PERC  Начальный процент трафика (по умолчанию: 10)"
            echo "  --canary-duration SEC     Длительность каждой фазы (по умолчанию: 300)"
            echo "  --max-percentage PERC     Максимальный процент трафика (по умолчанию: 50)"
            echo "  --use-istio BOOL          Использовать Istio (true/false)"
            echo ""
            echo "Пример:"
            echo "  $0 --image-tag v1.2.3 --canary-percentage 10 --canary-duration 600"
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