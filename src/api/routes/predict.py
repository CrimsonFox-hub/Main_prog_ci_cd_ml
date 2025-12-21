"""
Маршруты для предсказаний
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import time
import uuid
import asyncio

from src.ml_pipeline.inference.predictor import ModelPredictor
from src.utils.logger import api_logger, RequestContext
from src.utils.database import get_database_manager
from src.api.middleware.auth import require_auth, require_permission

router = APIRouter(prefix="/predict", tags=["predictions"])

# Модели Pydantic для валидации запросов
class FeatureData(BaseModel):
    """Модель данных для предсказания"""
    age: int = Field(..., ge=18, le=100, description="Возраст клиента")
    income: float = Field(..., ge=0, description="Годовой доход")
    credit_score: int = Field(..., ge=300, le=850, description="Кредитный скоринг")
    loan_amount: float = Field(..., ge=0, description="Сумма кредита")
    employment_years: int = Field(..., ge=0, description="Стаж работы")
    debt_to_income: Optional[float] = Field(None, ge=0, le=1, description="Отношение долга к доходу")
    has_default: Optional[bool] = Field(False, description="Были ли дефолты")
    loan_purpose: Optional[str] = Field(None, description="Цель кредита")
    
    @validator('debt_to_income')
    def validate_dti(cls, v):
        if v is not None and v > 5:  # Не реалистичное значение
            raise ValueError('Debt to income ratio too high')
        return v

class PredictionRequest(BaseModel):
    """Запрос на предсказание"""
    features: FeatureData
    model_version: Optional[str] = Field("latest", description="Версия модели")
    request_id: Optional[str] = Field(None, description="ID запроса для отслеживания")
    return_features: Optional[bool] = Field(False, description="Возвращать ли признаки в ответе")
    return_confidence: Optional[bool] = Field(True, description="Возвращать ли уверенность предсказания")

class BatchPredictionRequest(BaseModel):
    """Пакетный запрос на предсказание"""
    requests: List[PredictionRequest] = Field(..., max_items=1000)
    batch_id: Optional[str] = Field(None, description="ID батча для отслеживания")

class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    prediction: float = Field(..., ge=0, le=1, description="Вероятность дефолта")
    prediction_class: int = Field(..., ge=0, le=1, description="Класс предсказания (0=хороший, 1=плохой)")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Уверенность предсказания")
    model_version: str = Field(..., description="Версия модели")
    request_id: str = Field(..., description="ID запроса")
    processing_time_ms: float = Field(..., description="Время обработки в мс")
    features: Optional[Dict[str, Any]] = Field(None, description="Исходные признаки")
    threshold: float = Field(0.5, description="Порог классификации")

class BatchPredictionResponse(BaseModel):
    """Ответ на пакетный запрос"""
    predictions: List[PredictionResponse]
    batch_id: str = Field(..., description="ID батча")
    total_processed: int = Field(..., description="Всего обработано запросов")
    successful: int = Field(..., description="Успешных предсказаний")
    failed: int = Field(..., description="Неудачных предсказаний")
    total_time_ms: float = Field(..., description="Общее время обработки")

# Инициализация предсказателя
predictor = ModelPredictor()

@router.post("/single", response_model=PredictionResponse)
@require_auth()
@require_permission(["predict:single"])
async def predict_single(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """Предсказание для одного клиента"""
    start_time = time.time()
    
    # Генерация ID запроса, если не предоставлен
    request_id = request.request_id or str(uuid.uuid4())
    
    with RequestContext(
        request_id=request_id,
        correlation_id=request_id,
        user_id="api_user"  # Можно получить из аутентификации
    ):
        try:
            api_logger.info(
                f"Processing single prediction request",
                extra={
                    "request_id": request_id,
                    "model_version": request.model_version,
                    "features_count": len(request.features.dict())
                }
            )
            
            # Преобразование признаков
            features_dict = request.features.dict()
            
            # Предсказание
            prediction_result = predictor.predict(
                features=features_dict,
                model_version=request.model_version
            )
            
            # Расчет времени обработки
            processing_time = (time.time() - start_time) * 1000
            
            # Формирование ответа
            response = PredictionResponse(
                prediction=prediction_result["probability"],
                prediction_class=prediction_result["class"],
                confidence=prediction_result.get("confidence"),
                model_version=prediction_result["model_version"],
                request_id=request_id,
                processing_time_ms=processing_time,
                features=features_dict if request.return_features else None,
                threshold=prediction_result.get("threshold", 0.5)
            )
            
            # Логирование предсказания
            api_logger.info(
                f"Prediction completed",
                extra={
                    "request_id": request_id,
                    "prediction": prediction_result["probability"],
                    "prediction_class": prediction_result["class"],
                    "processing_time_ms": processing_time
                }
            )
            
            # Фоновая задача: сохранение в БД
            background_tasks.add_task(
                save_prediction_to_db,
                request_id=request_id,
                features=features_dict,
                prediction_result=prediction_result,
                processing_time=processing_time
            )
            
            return response
            
        except Exception as e:
            api_logger.error(
                f"Prediction failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "features": request.features.dict()
                },
                exc_info=True
            )
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Prediction failed",
                    "request_id": request_id,
                    "message": str(e)
                }
            )

@router.post("/batch", response_model=BatchPredictionResponse)
@require_auth()
@require_permission(["predict:batch"])
async def predict_batch(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
) -> BatchPredictionResponse:
    """Пакетное предсказание для множества клиентов"""
    start_time = time.time()
    
    # Генерация ID батча
    batch_id = batch_request.batch_id or str(uuid.uuid4())
    
    with RequestContext(
        request_id=batch_id,
        correlation_id=batch_id,
        user_id="api_user"
    ):
        try:
            api_logger.info(
                f"Processing batch prediction",
                extra={
                    "batch_id": batch_id,
                    "requests_count": len(batch_request.requests)
                }
            )
            
            predictions = []
            successful = 0
            failed = 0
            
            # Обработка запросов параллельно
            async def process_single_request(req: PredictionRequest, idx: int):
                nonlocal successful, failed
                
                req_id = req.request_id or f"{batch_id}_{idx}"
                
                try:
                    # Предсказание
                    prediction_result = predictor.predict(
                        features=req.features.dict(),
                        model_version=req.model_version
                    )
                    
                    # Формирование ответа
                    response = PredictionResponse(
                        prediction=prediction_result["probability"],
                        prediction_class=prediction_result["class"],
                        confidence=prediction_result.get("confidence"),
                        model_version=prediction_result["model_version"],
                        request_id=req_id,
                        processing_time_ms=0,  # Заполнится позже
                        features=req.features.dict() if req.return_features else None,
                        threshold=prediction_result.get("threshold", 0.5)
                    )
                    
                    predictions.append(response)
                    successful += 1
                    
                    # Фоновая задача: сохранение в БД
                    background_tasks.add_task(
                        save_prediction_to_db,
                        request_id=req_id,
                        features=req.features.dict(),
                        prediction_result=prediction_result,
                        processing_time=0
                    )
                    
                except Exception as e:
                    api_logger.error(
                        f"Batch prediction item failed",
                        extra={
                            "batch_id": batch_id,
                            "request_index": idx,
                            "error": str(e)
                        }
                    )
                    failed += 1
            
            # Параллельная обработка
            tasks = [
                process_single_request(req, idx)
                for idx, req in enumerate(batch_request.requests)
            ]
            
            await asyncio.gather(*tasks)
            
            # Расчет общего времени
            total_time = (time.time() - start_time) * 1000
            
            api_logger.info(
                f"Batch prediction completed",
                extra={
                    "batch_id": batch_id,
                    "total_processed": len(batch_request.requests),
                    "successful": successful,
                    "failed": failed,
                    "total_time_ms": total_time
                }
            )
            
            return BatchPredictionResponse(
                predictions=predictions,
                batch_id=batch_id,
                total_processed=len(batch_request.requests),
                successful=successful,
                failed=failed,
                total_time_ms=total_time
            )
            
        except Exception as e:
            api_logger.error(
                f"Batch prediction failed",
                extra={
                    "batch_id": batch_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Batch prediction failed",
                    "batch_id": batch_id,
                    "message": str(e)
                }
            )

@router.get("/explain/{request_id}")
@require_auth()
@require_permission(["predict:explain"])
async def explain_prediction(request_id: str):
    """Объяснение предсказания (SHAP значения)"""
    try:
        # Получение записи из БД
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        query = """
        SELECT features, prediction_result
        FROM predictions
        WHERE request_id = %s
        """
        
        result = db_pool.execute_query(query, (request_id,))
        
        if not result:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Генерация SHAP объяснений
        features = result[0]["features"]
        explanation = predictor.explain_prediction(features)
        
        return {
            "request_id": request_id,
            "explanation": explanation,
            "features": features
        }
        
    except Exception as e:
        api_logger.error(
            f"Explanation failed for request {request_id}",
            extra={"error": str(e)},
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Explanation generation failed",
                "message": str(e)
            }
        )

@router.get("/statistics")
@require_auth()
@require_permission(["predict:stats"])
async def get_prediction_statistics(
    hours: int = 24,
    model_version: Optional[str] = None
):
    """Статистика предсказаний за период"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        # Основная статистика
        stats_query = """
        SELECT 
            COUNT(*) as total_predictions,
            AVG(processing_time_ms) as avg_processing_time,
            MIN(processing_time_ms) as min_processing_time,
            MAX(processing_time_ms) as max_processing_time,
            AVG((prediction_result->>'probability')::float) as avg_probability,
            SUM(CASE WHEN (prediction_result->>'class')::int = 1 THEN 1 ELSE 0 END) as bad_predictions,
            SUM(CASE WHEN (prediction_result->>'class')::int = 0 THEN 1 ELSE 0 END) as good_predictions
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '%s hours'
        """
        
        params = [hours]
        if model_version:
            stats_query += " AND model_version = %s"
            params.append(model_version)
        
        stats = db_pool.execute_query(stats_query, tuple(params))[0]
        
        # Распределение по часам
        distribution_query = """
        SELECT 
            EXTRACT(HOUR FROM created_at) as hour,
            COUNT(*) as predictions_count
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '%s hours'
        GROUP BY hour
        ORDER BY hour
        """
        
        distribution = db_pool.execute_query(distribution_query, (hours,))
        
        return {
            "period_hours": hours,
            "model_version": model_version or "all",
            "statistics": stats,
            "hourly_distribution": distribution
        }
        
    except Exception as e:
        api_logger.error(
            f"Failed to get prediction statistics",
            extra={"error": str(e)},
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get statistics",
                "message": str(e)
            }
        )

async def save_prediction_to_db(
    request_id: str,
    features: Dict[str, Any],
    prediction_result: Dict[str, Any],
    processing_time: float
):
    """Фоновая задача для сохранения предсказания в БД"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        query = """
        INSERT INTO predictions (
            request_id,
            features,
            prediction_result,
            processing_time_ms,
            model_version,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, NOW())
        """
        
        db_pool.execute_query(
            query,
            (
                request_id,
                features,
                prediction_result,
                processing_time,
                prediction_result.get("model_version", "unknown")
            ),
            fetch=False
        )
        
        api_logger.debug(f"Prediction saved to DB: {request_id}")
        
    except Exception as e:
        api_logger.error(
            f"Failed to save prediction to DB: {request_id}",
            extra={"error": str(e)}
        )