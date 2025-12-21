# scripts/setup-minio.ps1
Write-Host "Setting up MinIO for MLflow..." -ForegroundColor Green

# Ждем MinIO
Write-Host "Waiting for MinIO to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Создаем bucket через Docker exec
Write-Host "Creating MLflow bucket..." -ForegroundColor Yellow

# Используем MinIO клиент внутри контейнера
docker exec credit_scoring_minio sh -c "
    mc alias set myminio http://localhost:9000 minioadmin minioadmin
    mc mb myminio/mlflow-artifacts --ignore-existing
    mc anonymous set public myminio/mlflow-artifacts
    echo 'Bucket created successfully'
"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ MinIO bucket 'mlflow-artifacts' created" -ForegroundColor Green
} else {
    Write-Host "⚠️ Could not create bucket. You may need to create it manually:" -ForegroundColor Yellow
    Write-Host "1. Open http://localhost:9001" -ForegroundColor White
    Write-Host "2. Login with minioadmin/minioadmin" -ForegroundColor White
    Write-Host "3. Create bucket named 'mlflow-artifacts'" -ForegroundColor White
}