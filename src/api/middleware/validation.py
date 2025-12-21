"""
Middleware для валидации запросов
"""
import json
import jsonschema
from typing import Dict, Any
from fastapi import Request, HTTPException, status
from pydantic import BaseModel, ValidationError

from src.utils.logger import api_logger

class RequestValidator:
    """Класс для валидации входящих запросов"""
    
    def __init__(self):
        # Схемы валидации для разных эндпоинтов
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Загрузка JSON схем для валидации"""
        return {
            "/api/v1/predict": {
                "type": "object",
                "required": ["features"],
                "properties": {
                    "features": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "model_version": {
                        "type": "string",
                        "pattern": "^v\\d+\\.\\d+\\.\\d+$"
                    },
                    "request_id": {
                        "type": "string",
                        "minLength": 1
                    }
                }
            },
            "/api/v1/batch_predict": {
                "type": "object",
                "required": ["requests"],
                "properties": {
                    "requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["features"],
                            "properties": {
                                "features": {
                                    "type": "object",
                                    "additionalProperties": True
                                }
                            }
                        },
                        "maxItems": 1000
                    }
                }
            },
            "/api/v1/models/register": {
                "type": "object",
                "required": ["model_name", "model_path", "version"],
                "properties": {
                    "model_name": {"type": "string"},
                    "model_path": {"type": "string"},
                    "version": {"type": "string"},
                    "metadata": {"type": "object"}
                }
            }
        }
    
    async def validate_request(self, request: Request):
        """Валидация входящего запроса"""
        path = request.url.path
        
        # Проверка, есть ли схема для этого пути
        if path not in self.schemas:
            return
        
        # Чтение тела запроса
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in request body"
            )
        
        # Валидация по JSON схеме
        schema = self.schemas[path]
        
        try:
            jsonschema.validate(instance=body, schema=schema)
        except jsonschema.ValidationError as e:
            api_logger.warning(
                f"Schema validation failed for {path}",
                extra={
                    "error": str(e),
                    "schema": schema,
                    "body": body
                }
            )
            
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Schema validation failed",
                    "message": e.message,
                    "path": list(e.path)
                }
            )
        
        # Дополнительная бизнес-логика валидации
        await self._validate_business_logic(path, body, request)
    
    async def _validate_business_logic(self, path: str, body: Dict[str, Any], request: Request):
        """Дополнительная бизнес-логика валидации"""
        if path == "/api/v1/predict":
            # Проверка наличия обязательных фич
            required_features = ["age", "income", "credit_score"]
            features = body.get("features", {})
            
            missing_features = [
                feature for feature in required_features
                if feature not in features
            ]
            
            if missing_features:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required features: {missing_features}"
                )
            
            # Валидация диапазонов значений
            age = features.get("age")
            if age and (age < 18 or age > 100):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Age must be between 18 and 100"
                )
            
            income = features.get("income")
            if income and income < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Income cannot be negative"
                )

class PydanticValidationMiddleware:
    """Middleware для валидации с использованием Pydantic"""
    
    def __init__(self, models: Dict[str, BaseModel]):
        self.models = models  # Маппинг путь -> Pydantic модель
    
    async def __call__(self, request: Request, call_next):
        path = request.url.path
        
        # Проверка, есть ли Pydantic модель для этого пути
        if path in self.models:
            model_class = self.models[path]
            
            try:
                # Чтение и валидация тела запроса
                body = await request.json()
                validated_data = model_class(**body)
                
                # Замена тела запроса на валидированные данные
                request.state.validated_data = validated_data
                
            except ValidationError as e:
                api_logger.warning(
                    f"Pydantic validation failed for {path}",
                    extra={
                        "errors": e.errors(),
                        "body": body
                    }
                )
                
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "Validation error",
                        "errors": e.errors()
                    }
                )
        
        return await call_next(request)