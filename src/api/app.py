"""
FastAPI приложение для кредитного скоринга с ONNX моделью
"""
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.ml_pipeline.inference.predictor import get_predictor
from src.ml_pipeline.inference.explainer import get_explainer, init_explainer

import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime
import time
import psutil
import platform

from src.ml_pipeline.inference.predictor import ModelPredictor
from src.ml_pipeline.inference.preprocessor import FeaturePreprocessor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="Credit Scoring API",
    description="API для оценки кредитоспособности заемщиков с ONNX моделью",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")
templates = Jinja2Templates(directory="src/api/templates")

# Глобальные объекты
model_predictor = None
feature_preprocessor = None
service_links = {}

def load_model_and_explainer():
    """Загрузка модели и инициализация объяснителя"""
    try:
        # Загрузка предсказателя
        predictor = get_predictor("models/trained/model.pkl")
        
        # Заглушка для теста - в реальном проекте загрузите реальные данные
        feature_names = [
            "checking_status", "duration", "credit_history", "purpose", 
            "credit_amount", "savings_status", "employment", "installment_commitment",
            "personal_status", "other_parties", "residence_since", "property_magnitude",
            "age", "other_payment_plans", "housing", "existing_credits",
            "job", "num_dependents", "own_telephone", "foreign_worker"
        ]
        
        # Инициализация объяснителя (заглушка для теста)
        explainer = get_explainer()
        
        logger.info("Model and explainer initialized")
        return predictor, explainer, feature_names
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, []

# Загружаем при старте
predictor, explainer, feature_names = load_model_and_explainer()
# Модели данных
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_loaded: bool
    services: Dict[str, str]

class FeatureRequest(BaseModel):
    age: int = Field(25, ge=18, le=100, description="Возраст заемщика")
    income: float = Field(50000, ge=0, le=1000000, description="Годовой доход")
    credit_score: int = Field(650, ge=300, le=850, description="Кредитный рейтинг")
    loan_amount: float = Field(10000, ge=1000, le=100000, description="Сумма кредита")
    employment_years: int = Field(5, ge=0, le=50, description="Стаж работы (лет)")
    debt_to_income: float = Field(0.3, ge=0, le=1, description="Отношение долга к доходу")
    has_default: bool = Field(False, description="Наличие дефолта в истории")
    request_id: Optional[str] = None
    loan_purpose: str = Field("home", description="Цель кредита", 
                              regex="^(car|home|education|business|other)$")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    request_id: str
    timestamp: str
    model_version: str = "1.0.0"
    explanation: Optional[Dict[str, Any]] = None
    risk_level: Optional[str] = Field(default="medium", description="Уровень риска")
    recommendation: Optional[str] = Field(default="approve", description="Рекомендация")
    processing_time_ms: Optional[float] = Field(default=0.0, description="Время обработки в мс")

class BatchRequest(BaseModel):
    requests: List[FeatureRequest]

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_id: str
    total_approved: int
    total_rejected: int
    approval_rate: float

# Инициализация при запуске
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    global model_predictor, feature_preprocessor, service_links
    
    logger.info("Initializing Credit Scoring API...")
    
    try:
        # Инициализация препроцессора
        feature_preprocessor = FeaturePreprocessor()
        logger.info("Feature preprocessor initialized")
        
        # Инициализация предсказателя
        model_path = os.getenv("MODEL_PATH", "models/credit_scoring.onnx")
        model_predictor = ModelPredictor(model_path)
        logger.info(f"Model predictor initialized with model: {model_path}")
        
        # Информация о сервисах
        service_links = {
            "api_docs": "http://localhost:8000/api/docs",
            "mlflow": "http://localhost:5000",
            "grafana": "http://localhost:3000",
            "minio": "http://localhost:9001",
            "loki": "http://localhost:3100",
            "prometheus": "http://localhost:9090",
            "postgres": "localhost:5432",
            "redis": "localhost:6379",
            "kibana": "http://localhost:5601"  # если используется
        }
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}", exc_info=True)
        raise

# HTML страницы
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Главная страница с демонстрацией"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "services": service_links,
        "model_info": model_predictor.get_model_info() if model_predictor else {}
    })

@app.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request):
    """Страница демонстрации модели"""
    return templates.TemplateResponse("demo.html", {
        "request": request,
        "model_info": model_predictor.get_model_info() if model_predictor else {}
    })

@app.get("/services", include_in_schema=False)
async def services_status():
    """Статус сервисов (заглушка)"""
    return {
        "api": "running",
        "database": "connected",
        "mlflow": "available",
        "redis": "connected",
        "minio": "available"
    }

@app.get("/monitoring", include_in_schema=False)
async def monitoring_links():
    """Ссылки на мониторинг (заглушка)"""
    return {
        "grafana": "http://localhost:3000",
        "prometheus": "http://localhost:9090",
        "loki": "http://localhost:3100"
    }

# API эндпоинты
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/ping")
async def ping():
    """Простой эндпоинт для проверки доступности"""
    return {"message": "pong", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_single(request: FeatureRequest):
    """Предсказание для одного клиента"""
    try:
        # Преобразование признаков в список
        features = [
            request.age,
            request.income,
            request.credit_score,
            request.loan_amount,
            request.employment_years,
            request.debt_to_income,
            float(request.has_default),
            1.0 if request.loan_purpose == "car" else 0.0,
            1.0 if request.loan_purpose == "home" else 0.0,
            1.0 if request.loan_purpose == "education" else 0.0,
            1.0 if request.loan_purpose == "business" else 0.0,
        ]
        
        # Логирование
        logger.info(f"Prediction request: {request.request_id}")
        
        # Временная заглушка
        prediction = 1
        probability = 0.85
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            request_id=request.request_id or "default",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch_predict", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Пакетное предсказание для нескольких клиентов"""
    if not model_predictor or not model_predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if len(request.requests) == 0:
            raise HTTPException(status_code=400, detail="No requests provided")
        
        if len(request.requests) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 requests per batch")
        
        predictions = []
        total_approved = 0
        
        for req in request.requests:
            pred_response = await predict_single(req)
            predictions.append(pred_response)
            
            if pred_response.prediction == 1:
                total_approved += 1
        
        approval_rate = total_approved / len(request.requests)
        
        return BatchResponse(
            predictions=predictions,
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_approved=total_approved,
            total_rejected=len(request.requests) - total_approved,
            approval_rate=approval_rate
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/info")
async def get_model_info():
    """Получение информации о модели"""
    if not model_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_info": model_predictor.get_model_info(),
        "performance_stats": model_predictor.get_performance_stats(),
        "preprocessor_info": {
            "numerical_features": feature_preprocessor.numerical_features if feature_preprocessor else [],
            "categorical_features": feature_preprocessor.categorical_features if feature_preprocessor else []
        }
    }

@app.get("/api/v1/system/metrics")
async def get_system_metrics():
    """Метрики системы для Prometheus"""
    import psutil
    
    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "python_version": platform.python_version(),
            "process_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
        },
        "api": {
            "status": "running",
            "model_loaded": model_predictor is not None and model_predictor.model_loaded,
            "cache_size": model_predictor.get_performance_stats()['cache_hits'] if model_predictor else 0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/services")
async def get_services():
    """Получение списка сервисов"""
    return {
        "services": service_links,
        "status": "available",
        "timestamp": datetime.now().isoformat()
    }

# Информационные эндпоинты
@app.get("/api/v1/features/schema")
async def get_feature_schema():
    """Схема признаков для UI"""
    return {
        "features": [
            {
                "name": "age",
                "type": "integer",
                "min": 18,
                "max": 100,
                "description": "Возраст заемщика",
                "default": 25
            },
            {
                "name": "income",
                "type": "float",
                "min": 0,
                "max": 1000000,
                "description": "Годовой доход",
                "default": 50000
            },
            {
                "name": "credit_score",
                "type": "integer",
                "min": 300,
                "max": 850,
                "description": "Кредитный рейтинг",
                "default": 650
            },
            {
                "name": "loan_amount",
                "type": "float",
                "min": 1000,
                "max": 100000,
                "description": "Сумма кредита",
                "default": 10000
            },
            {
                "name": "employment_years",
                "type": "integer",
                "min": 0,
                "max": 50,
                "description": "Стаж работы (лет)",
                "default": 5
            },
            {
                "name": "debt_to_income",
                "type": "float",
                "min": 0,
                "max": 1,
                "description": "Отношение долга к доходу",
                "default": 0.3
            },
            {
                "name": "has_default",
                "type": "boolean",
                "description": "Наличие дефолта в истории",
                "default": False
            },
            {
                "name": "loan_purpose",
                "type": "categorical",
                "options": ["car", "home", "education", "business", "other"],
                "description": "Цель кредита",
                "default": "home"
            }
        ],
        "target": {
            "name": "approval",
            "type": "binary",
            "description": "Одобрение кредита (1 - одобрено, 0 - отклонено)"
        }
    }

# Эндпоинт для тестирования модели
@app.get("/api/v1/test/random")
async def test_random_prediction():
    """Тестовый запрос со случайными данными"""
    import random
    
    test_request = FeatureRequest(
        age=random.randint(20, 60),
        income=random.uniform(30000, 150000),
        credit_score=random.randint(500, 800),
        loan_amount=random.uniform(5000, 50000),
        employment_years=random.randint(1, 30),
        debt_to_income=random.uniform(0.1, 0.6),
        has_default=random.choice([True, False]),
        loan_purpose=random.choice(["car", "home", "education", "business", "other"])
    )
    
    return await predict_single(test_request)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )