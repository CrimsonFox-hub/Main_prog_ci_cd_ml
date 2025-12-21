# Проверка здоровья сервисов для Windows
Write-Host "Проверка здоровья сервисов MLOps Credit Scoring" -ForegroundColor Green

$services = @(
    @{Name="API"; Port=8000; Path="/health"},
    @{Name="MLflow"; Port=5000; Path=""},
    @{Name="MinIO Console"; Port=9001; Path=""},
    @{Name="Grafana"; Port=3000; Path=""},
    @{Name="Prometheus"; Port=9090; Path=""},
    @{Name="Loki"; Port=3100; Path="/ready"}
)

foreach ($service in $services) {
    $url = "http://localhost:$($service.Port)$($service.Path)"
    try {
        $response = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 5 -ErrorAction Stop
        Write-Host "✓ $($service.Name) доступен ($url)" -ForegroundColor Green
    } catch {
        Write-Host "✗ $($service.Name) недоступен ($url)" -ForegroundColor Red
    }
}

# Проверка Docker контейнеров
Write-Host "`nПроверка Docker контейнеров..." -ForegroundColor Yellow
docker-compose ps
