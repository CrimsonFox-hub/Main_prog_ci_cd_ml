Write-Host "==========================================" -ForegroundColor Green
Write-Host "ЗАПУСК МОНИТОРИНГА CREDIT SCORING" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Создаем директории для Grafana
Write-Host "`nСоздание директорий для Grafana..." -ForegroundColor Yellow

# Создаем структуру директорий
$dirs = @(
    "configs\grafana\datasources",
    "configs\grafana\dashboards",
    "configs\prometheus"
)

foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Создана директория: $dir" -ForegroundColor White
    }
}

# Создаем конфигурационные файлы
Write-Host "`nСоздание конфигураций..." -ForegroundColor Yellow

# Datasources для Grafana
$datasourcesContent = @'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
'@

$datasourcesContent | Out-File -FilePath "configs\grafana\datasources\datasources.yml" -Encoding UTF8
Write-Host "  Создан datasources.yml" -ForegroundColor Green

# Dashboards для Grafana
$dashboardsContent = @'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/provisioning/dashboards
'@

$dashboardsContent | Out-File -FilePath "configs\grafana\dashboards\dashboards.yml" -Encoding UTF8
Write-Host "  Создан dashboards.yml" -ForegroundColor Green

# Prometheus config (основной)
$prometheusContent = @'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: "credit-scoring"

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
        labels:
          service: "prometheus"

  - job_name: "api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: "/metrics"
    scrape_interval: 10s
    scrape_timeout: 5s
    labels:
      service: "api"
      environment: "development"

  - job_name: "mlflow"
    static_configs:
      - targets: ["mlflow:5000"]
    metrics_path: "/metrics"
    labels:
      service: "mlflow"
      environment: "development"
'@

$prometheusContent | Out-File -FilePath "configs\prometheus\prometheus.yml" -Encoding UTF8
Write-Host "  Создан prometheus.yml" -ForegroundColor Green

# Запускаем сервисы мониторинга
Write-Host "`nЗапуск сервисов мониторинга..." -ForegroundColor Yellow
docker-compose up -d prometheus loki grafana

Write-Host "`nОжидание запуска (15 секунд)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Проверка доступности
Write-Host "`nПроверка доступности:" -ForegroundColor Yellow

$services = @(
    @{Name="Prometheus"; Port=9090},
    @{Name="Loki"; Port=3100},
    @{Name="Grafana"; Port=3000}
)

foreach ($service in $services) {
    try {
        $result = Test-NetConnection -ComputerName localhost -Port $service.Port -WarningAction SilentlyContinue
        if ($result.TcpTestSucceeded) {
            Write-Host "  $($service.Name) доступен" -ForegroundColor Green
        } else {
            Write-Host "  $($service.Name) недоступен" -ForegroundColor Red
        }
    } catch {
        Write-Host "  $($service.Name) ошибка проверки" -ForegroundColor Red
    }
}

Write-Host "`nДоступ к сервисам:" -ForegroundColor Yellow
Write-Host "  Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "  Grafana:    http://localhost:3000 (логин: admin/admin)" -ForegroundColor White
Write-Host "  Loki:       http://localhost:3100" -ForegroundColor White

Write-Host "`nДля настройки Grafana:" -ForegroundColor Yellow
Write-Host "  1. Откройте http://localhost:3000" -ForegroundColor White
Write-Host "  2. Войдите с логином admin / admin" -ForegroundColor White
Write-Host "  3. Добавьте источник данных Prometheus: http://prometheus:9090" -ForegroundColor White

Write-Host "`nДля остановки мониторинга:" -ForegroundColor Yellow
Write-Host "  docker-compose stop prometheus loki grafana" -ForegroundColor White
