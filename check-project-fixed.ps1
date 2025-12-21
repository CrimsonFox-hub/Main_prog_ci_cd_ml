# check-project.ps1 - Проверка работоспособности проекта
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MLOps CREDIT SCORING - ПРОВЕРКА СИСТЕМЫ" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Проверка Docker
Write-Host "1. Проверка Docker..." -ForegroundColor Yellow
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $dockerVersion = docker --version
    Write-Host "   ✓ Docker: $dockerVersion" -ForegroundColor Green
} else {
    Write-Host "   ✗ Docker не установлен" -ForegroundColor Red
    exit 1
}

if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
    Write-Host "   ✓ Docker Compose доступен" -ForegroundColor Green
} else {
    Write-Host "   ✗ Docker Compose не установлен" -ForegroundColor Red
    exit 1
}

# 2. Проверка контейнеров
Write-Host "`n2. Состояние контейнеров:" -ForegroundColor Yellow
docker-compose ps

# 3. Проверка портов
Write-Host "`n3. Проверка доступности портов:" -ForegroundColor Yellow
$services = @(
    @{Name="PostgreSQL"; Port=5432; Required=$true},
    @{Name="Redis"; Port=6379; Required=$true},
    @{Name="MinIO API"; Port=9000; Required=$true},
    @{Name="MinIO Console"; Port=9001; Required=$true},
    @{Name="MLflow"; Port=5000; Required=$true},
    @{Name="API"; Port=8000; Required=$true},
    @{Name="Prometheus"; Port=9090; Required=$false},
    @{Name="Grafana"; Port=3000; Required=$false},
    @{Name="Loki"; Port=3100; Required=$false}
)

foreach ($service in $services) {
    $result = Test-NetConnection -ComputerName localhost -Port $service.Port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
    $status = if ($result.TcpTestSucceeded) { "✓ ОТКРЫТ" } else { "✗ ЗАКРЫТ" }
    $color = if ($result.TcpTestSucceeded) { "Green" } else { if ($service.Required) { "Red" } else { "Yellow" } }
    
    Write-Host "   $status - $($service.Name): порт $($service.Port)" -ForegroundColor $color
}

# 4. Проверка API
Write-Host "`n4. Проверка API:" -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5 -ErrorAction Stop
    Write-Host "   ✓ API работает: $($healthResponse.status)" -ForegroundColor Green
    
    # Проверка основных эндпоинтов
    $endpoints = @(
        @{Path="/"; Method="GET"},
        @{Path="/docs"; Method="GET"},
        @{Path="/api/v1/predict"; Method="POST"}
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            if ($endpoint.Method -eq "GET") {
                $response = Invoke-WebRequest -Uri "http://localhost:8000$($endpoint.Path)" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
                if ($response.StatusCode -in @(200, 201, 301, 302)) {
                    Write-Host "     ✓ $($endpoint.Path) доступен" -ForegroundColor Green
                }
            }
        } catch {
            Write-Host "     ✗ $($endpoint.Path) недоступен" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "   ✗ API недоступен: $($_.Exception.Message)" -ForegroundColor Red
}

# 5. Проверка MLflow
Write-Host "`n5. Проверка MLflow:" -ForegroundColor Yellow
try {
    $mlflowResponse = Invoke-WebRequest -Uri "http://localhost:5000" -Method Get -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($mlflowResponse.StatusCode -eq 200) {
        Write-Host "   ✓ MLflow доступен" -ForegroundColor Green
    } else {
        Write-Host " MLflow отвечает с кодом: $($mlflowResponse.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ✗ MLflow недоступен" -ForegroundColor Red
    Write-Host "     Проверьте логи: docker-compose logs mlflow" -ForegroundColor Gray
}

# 6. Проверка MinIO
Write-Host "`n6. Проверка MinIO:" -ForegroundColor Yellow
try {
    $minioResponse = Invoke-WebRequest -Uri "http://localhost:9001" -Method Get -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($minioResponse.StatusCode -eq 200) {
        Write-Host "   ✓ MinIO Console доступен" -ForegroundColor Green
        Write-Host "     Логин: minioadmin, Пароль: minioadmin" -ForegroundColor Gray
    } else {
        Write-Host "MinIO отвечает с кодом: $($minioResponse.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ✗ MinIO недоступен" -ForegroundColor Red
}

# 7. Проверка данных и моделей
Write-Host "`n7. Проверка файловой структуры:" -ForegroundColor Yellow
$directories = @("data/raw", "data/processed", "models/trained", "models/onnx", "logs", "configs")
foreach ($dir in $directories) {
    if (Test-Path $dir) {
        $itemCount = (Get-ChildItem $dir -Recurse -File 2>$null | Measure-Object).Count
        Write-Host "   ✓ $dir существует (файлов: $itemCount)" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $dir отсутствует" -ForegroundColor Red
    }
}

# 8. Общая статистика
Write-Host "`n8. Общая статистика:" -ForegroundColor Yellow
$containerCount = (docker ps -q 2>$null | Measure-Object).Count
$imageCount = (docker images -q 2>$null | Measure-Object).Count
$volumeCount = (docker volume ls -q 2>$null | Measure-Object).Count

Write-Host "   Контейнеры запущены: $containerCount" -ForegroundColor Cyan
Write-Host "   Docker образов: $imageCount" -ForegroundColor Cyan
Write-Host "   Docker томов: $volumeCount" -ForegroundColor Cyan

# 9. Рекомендации
Write-Host "`n9. Рекомендации:" -ForegroundColor Yellow

if ($containerCount -lt 5) {
    Write-Host " Запущено мало контейнеров. Запустите: .\run-project.ps1" -ForegroundColor Yellow
}

try {
    $apiHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($apiHealth.status -eq "healthy") {
        Write-Host "   ✓ API работает корректно" -ForegroundColor Green
        Write-Host "     Тестовый запрос: curl -X POST http://localhost:8000/api/v1/predict " -ForegroundColor Gray
        Write-Host "       -H 'Content-Type: application/json' " -ForegroundColor Gray
        Write-Host "       -d '{\""features\"": [0.1, 0.2, 0.3, 0.4, 0.5]}'" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ✗ API не отвечает" -ForegroundColor Red
}

# 10. Сводка
Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "СВОДКА ПРОВЕРКИ:" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$requiredServices = @("PostgreSQL", "Redis", "MinIO", "MLflow", "API")
$allOk = $true

foreach ($service in $requiredServices) {
    $port = switch ($service) {
        "PostgreSQL" { 5432 }
        "Redis" { 6379 }
        "MinIO" { 9001 }
        "MLflow" { 5000 }
        "API" { 8000 }
    }
    
    $result = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
    if ($result.TcpTestSucceeded) {
        Write-Host "✓ $service - РАБОТАЕТ" -ForegroundColor Green
    } else {
        Write-Host "✗ $service - НЕ РАБОТАЕТ" -ForegroundColor Red
        $allOk = $false
    }
}

if ($allOk) {
    Write-Host "`n ВСЕ ОСНОВНЫЕ СЕРВИСЫ РАБОТАЮТ КОРРЕКТНО!" -ForegroundColor Green
    Write-Host "   Проект готов к работе." -ForegroundColor Green
} else {
    Write-Host "`n НЕКОТОРЫЕ СЕРВИСЫ НЕ РАБОТАЮТ" -ForegroundColor Yellow
    Write-Host "   Для исправления запустите: .\run-project.ps1" -ForegroundColor White
}

Write-Host "`nДля подробной информации:" -ForegroundColor Gray
Write-Host "  • Логи API: docker-compose logs api" -ForegroundColor White
Write-Host "  • Логи MLflow: docker-compose logs mlflow" -ForegroundColor White
Write-Host "  • Остановка: docker-compose down" -ForegroundColor White
