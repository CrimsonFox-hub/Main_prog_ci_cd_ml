Write-Host "==========================================" -ForegroundColor Green
Write-Host "ЗАПУСК CREDIT SCORING SYSTEM" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Шаг 1: Остановка всего
Write-Host "`nШаг 1: Остановка предыдущих запусков..." -ForegroundColor Yellow
docker-compose down

# Шаг 2: Проверка необходимых файлов
Write-Host "`nШаг 2: Проверка файлов..." -ForegroundColor Yellow
if (!(Test-Path "docker\mlflow\Dockerfile")) {
    Write-Host "  ✗ Файл docker\mlflow\Dockerfile не найден" -ForegroundColor Red
    exit 1
}

# Шаг 3: Запуск базовой инфраструктуры
Write-Host "`nШаг 3: Запуск базовой инфраструктуры..." -ForegroundColor Yellow
docker-compose up -d postgres redis minio

Write-Host "  Ожидание 10 секунд..." -ForegroundColor Gray
Start-Sleep -Seconds 10

# Шаг 4: Настройка MinIO
Write-Host "`nШаг 4: Настройка MinIO..." -ForegroundColor Yellow
try {
    docker exec credit_scoring_minio mc alias set myminio http://localhost:9000 minioadmin minioadmin 2>$null
    docker exec credit_scoring_minio mc mb myminio/mlflow-artifacts --ignore-existing 2>$null
    docker exec credit_scoring_minio mc anonymous set public myminio/mlflow-artifacts 2>$null
    Write-Host "  ✓ MinIO настроен" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ MinIO уже настроен или ошибка" -ForegroundColor Yellow
}

# Шаг 5: Сборка MLflow
Write-Host "`nШаг 5: Сборка MLflow..." -ForegroundColor Yellow
docker-compose build mlflow

# Шаг 6: Запуск MLflow
Write-Host "`nШаг 6: Запуск MLflow..." -ForegroundColor Yellow
docker-compose up -d mlflow

Write-Host "  Ожидание 5 секунд..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# Шаг 7: Запуск API
Write-Host "`nШаг 7: Запуск API..." -ForegroundColor Yellow
docker-compose up -d api

# Шаг 8: Проверка состояния
Write-Host "`nШаг 8: Проверка состояния..." -ForegroundColor Yellow
docker-compose ps

# Шаг 9: Опциональный запуск мониторинга
Write-Host "`nШаг 9: Запуск мониторинга (опционально)..." -ForegroundColor Yellow
$choice = Read-Host "Запустить мониторинг? (y/n)"
if ($choice -eq 'y') {
    & ".\run-monitoring.ps1"
}

# Финальная информация
Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "СИСТЕМА ЗАПУЩЕНА!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

Write-Host "`nОсновные сервисы:" -ForegroundColor Cyan
Write-Host "  API:      http://localhost:8000" -ForegroundColor White
Write-Host "  MLflow:   http://localhost:5000" -ForegroundColor White
Write-Host "  MinIO:    http://localhost:9001" -ForegroundColor White
Write-Host "    Логин: minioadmin" -ForegroundColor White
Write-Host "    Пароль: minioadmin" -ForegroundColor White

Write-Host "`nДля проверки API:" -ForegroundColor Cyan
Write-Host "  curl http://localhost:8000/health" -ForegroundColor White

Write-Host "`nДля просмотра логов:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f mlflow" -ForegroundColor White
Write-Host "  docker-compose logs -f api" -ForegroundColor White

Write-Host "`nДля остановки всех сервисов:" -ForegroundColor Cyan
Write-Host "  docker-compose down" -ForegroundColor White