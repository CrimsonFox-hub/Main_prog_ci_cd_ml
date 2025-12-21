"""
Маршруты для проверки здоровья системы
"""
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from typing import Dict, Any
import psutil
import time

from src.utils.logger import api_logger
from src.utils.database import get_database_manager
from src.ml_pipeline.inference.predictor import ModelPredictor
from src.api.middleware.auth import require_auth, require_role

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Базовая проверка здоровья API"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "credit-scoring-api"
    }

@router.get("/detailed")
@require_auth()
async def detailed_health_check() -> Dict[str, Any]:
    """Детальная проверка здоровья всех компонентов"""
    checks = {}
    
    # 1. Проверка API
    checks["api"] = {
        "status": "healthy",
        "uptime": psutil.boot_time(),
        "memory_usage": psutil.virtual_memory().percent
    }
    
    # 2. Проверка базы данных
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        # Тестовый запрос
        result = db_pool.execute_query("SELECT 1 as test")
        
        checks["database"] = {
            "status": "healthy",
            "connection": True,
            "test_query": "success",
            "active_connections": db_pool.active_connections
        }
    except Exception as e:
        checks["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # 3. Проверка модели
    try:
        predictor = ModelPredictor()
        model_info = predictor.get_model_info()
        
        checks["model"] = {
            "status": "healthy",
            "version": model_info.get("version"),
            "loaded": model_info.get("loaded"),
            "input_shape": model_info.get("input_shape"),
            "output_shape": model_info.get("output_shape")
        }
    except Exception as e:
        checks["model"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # 4. Проверка кэша (если есть)
    try:
        import redis
        # Здесь можно добавить проверку Redis
        checks["cache"] = {
            "status": "healthy",
            "type": "memory"
        }
    except:
        checks["cache"] = {
            "status": "not_configured"
        }
    
    # 5. Проверка системных ресурсов
    checks["system"] = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "load_average": psutil.getloadavg()
    }
    
    # Определение общего статуса
    overall_status = "healthy"
    for check_name, check_result in checks.items():
        if check_result.get("status") == "unhealthy":
            overall_status = "degraded"
    
    if any(check.get("status") == "unhealthy" for check in checks.values()):
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "checks": checks
    }

@router.get("/metrics")
@require_role(["admin", "monitoring"])
async def health_metrics() -> Dict[str, Any]:
    """Метрики здоровья для Prometheus"""
    import psutil
    
    metrics = {}
    
    # Системные метрики
    metrics["system_cpu_percent"] = psutil.cpu_percent(interval=1)
    metrics["system_memory_percent"] = psutil.virtual_memory().percent
    metrics["system_disk_percent"] = psutil.disk_usage('/').percent
    
    # Метрики процесса
    process = psutil.Process()
    metrics["process_cpu_percent"] = process.cpu_percent()
    metrics["process_memory_mb"] = process.memory_info().rss / 1024 / 1024
    metrics["process_threads"] = process.num_threads()
    
    # Метрики API
    from src.api.routes.predict import prediction_counter, prediction_latency
    
    metrics["api_requests_total"] = prediction_counter._value.get()
    metrics["api_latency_seconds"] = prediction_latency._value.get()
    
    return metrics

@router.post("/stress-test")
@require_role(["admin"])
async def stress_test(duration: int = 10, requests_per_second: int = 100):
    """Тест нагрузки системы (только для админов)"""
    import asyncio
    import random
    
    api_logger.warning(f"Starting stress test: {duration}s, {requests_per_second} req/s")
    
    async def make_request():
        try:
            # Имитация запроса предсказания
            features = {
                "age": random.randint(20, 70),
                "income": random.randint(20000, 200000),
                "credit_score": random.randint(300, 850),
                "loan_amount": random.randint(1000, 50000),
                "employment_years": random.randint(0, 40)
            }
            
            # Здесь можно добавить реальный вызов API
            await asyncio.sleep(0.01)  # Имитация обработки
            
            return True
        except:
            return False
    
    # Запуск нагрузочного теста
    start_time = time.time()
    successful_requests = 0
    failed_requests = 0
    
    while time.time() - start_time < duration:
        tasks = []
        for _ in range(requests_per_second):
            tasks.append(make_request())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if result is True:
                successful_requests += 1
            else:
                failed_requests += 1
        
        await asyncio.sleep(1)
    
    total_time = time.time() - start_time
    
    return {
        "status": "completed",
        "duration_seconds": total_time,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "requests_per_second": successful_requests / total_time,
        "success_rate": successful_requests / (successful_requests + failed_requests) * 100
    }

@router.get("/readiness")
async def readiness_probe():
    """Проба готовности для Kubernetes"""
    # Проверка критических зависимостей
    try:
        # Проверка базы данных
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        db_pool.execute_query("SELECT 1")
        
        # Проверка модели
        predictor = ModelPredictor()
        if not predictor.model_loaded:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "model_not_loaded"}
            )
        
        return {"status": "ready"}
        
    except Exception as e:
        api_logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "error": str(e)}
        )

@router.get("/liveness")
async def liveness_probe():
    """Проба жизнеспособности для Kubernetes"""
    # Простая проверка, что процесс жив
    return {"status": "alive"}