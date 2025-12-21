"""
Маршруты для управления моделями
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import shutil
import json
import yaml
from pathlib import Path

from src.ml_pipeline.inference.predictor import ModelPredictor
from src.utils.logger import api_logger
from src.api.middleware.auth import require_auth, require_role
from src.ml_pipeline.training.onnx_conversion import ModelConverter

router = APIRouter(prefix="/models", tags=["models"])

# Модели Pydantic
class ModelInfo(BaseModel):
    """Информация о модели"""
    name: str
    version: str
    path: str
    status: str
    input_shape: List[int]
    output_shape: List[int]
    framework: str
    created_at: str
    metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ModelVersion(BaseModel):
    """Версия модели"""
    version: str
    path: str
    is_active: bool
    performance: Dict[str, float]
    created_at: str

class ModelUpdateRequest(BaseModel):
    """Запрос на обновление модели"""
    version: str
    make_active: bool = False

class ModelRegisterRequest(BaseModel):
    """Запрос на регистрацию модели"""
    name: str
    version: str
    model_path: str
    metadata: Optional[Dict[str, Any]] = None

@router.get("/", response_model=List[ModelInfo])
@require_auth()
@require_permission(["models:read"])
async def list_models():
    """Получение списка всех моделей"""
    try:
        models_dir = Path("models")
        models = []
        
        # Поиск моделей в директориях
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Проверка наличия метаданных
                metadata_path = model_dir / "metadata.yaml"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    model_info = ModelInfo(
                        name=metadata.get("name", model_dir.name),
                        version=metadata.get("version", "1.0.0"),
                        path=str(model_dir),
                        status="active" if metadata.get("active", False) else "inactive",
                        input_shape=metadata.get("input_shape", []),
                        output_shape=metadata.get("output_shape", []),
                        framework=metadata.get("framework", "onnx"),
                        created_at=metadata.get("created_at", ""),
                        metrics=metadata.get("metrics"),
                        metadata=metadata
                    )
                    
                    models.append(model_info)
        
        return models
        
    except Exception as e:
        api_logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}", response_model=ModelInfo)
@require_auth()
@require_permission(["models:read"])
async def get_model(model_name: str):
    """Получение информации о конкретной модели"""
    try:
        model_path = Path("models") / model_name
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        metadata_path = model_path / "metadata.yaml"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Model metadata not found")
        
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        return ModelInfo(
            name=metadata.get("name", model_name),
            version=metadata.get("version", "1.0.0"),
            path=str(model_path),
            status="active" if metadata.get("active", False) else "inactive",
            input_shape=metadata.get("input_shape", []),
            output_shape=metadata.get("output_shape", []),
            framework=metadata.get("framework", "onnx"),
            created_at=metadata.get("created_at", ""),
            metrics=metadata.get("metrics"),
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get model {model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}/versions", response_model=List[ModelVersion])
@require_auth()
@require_permission(["models:read"])
async def get_model_versions(model_name: str):
    """Получение всех версий модели"""
    try:
        model_path = Path("models") / model_name
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        versions = []
        
        # Поиск версий в поддиректориях
        for version_dir in model_path.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.yaml"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    version_info = ModelVersion(
                        version=metadata.get("version", version_dir.name),
                        path=str(version_dir),
                        is_active=metadata.get("active", False),
                        performance=metadata.get("performance", {}),
                        created_at=metadata.get("created_at", "")
                    )
                    
                    versions.append(version_info)
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
        
    except Exception as e:
        api_logger.error(f"Failed to get model versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register", response_model=ModelInfo)
@require_auth()
@require_role(["admin", "ml_engineer"])
async def register_model(request: ModelRegisterRequest):
    """Регистрация новой модели"""
    try:
        model_path = Path(request.model_path)
        
        if not model_path.exists():
            raise HTTPException(status_code=400, detail="Model file not found")
        
        # Создание директории для модели
        model_dir = Path("models") / request.name / request.version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Копирование модели
        if model_path.is_file():
            shutil.copy2(model_path, model_dir / model_path.name)
        else:
            # Копирование всей директории
            shutil.copytree(model_path, model_dir, dirs_exist_ok=True)
        
        # Создание метаданных
        metadata = {
            "name": request.name,
            "version": request.version,
            "active": False,  # Новая модель не активна по умолчанию
            "framework": "onnx",
            "created_at": datetime.now().isoformat(),
            "path": str(model_dir),
            "metadata": request.metadata or {}
        }
        
        # Сохранение метаданных
        metadata_path = model_dir / "metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        api_logger.info(f"Model registered: {request.name} v{request.version}")
        
        return ModelInfo(**metadata)
        
    except Exception as e:
        api_logger.error(f"Failed to register model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{model_name}/activate")
@require_auth()
@require_role(["admin", "ml_engineer"])
async def activate_model(model_name: str, request: ModelUpdateRequest):
    """Активация версии модели"""
    try:
        model_dir = Path("models") / model_name
        version_dir = model_dir / request.version
        
        if not version_dir.exists():
            raise HTTPException(status_code=404, detail="Model version not found")
        
        # Деактивация всех версий
        for v_dir in model_dir.iterdir():
            if v_dir.is_dir():
                metadata_path = v_dir / "metadata.yaml"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    metadata["active"] = False
                    
                    with open(metadata_path, 'w') as f:
                        yaml.dump(metadata, f)
        
        # Активация выбранной версии
        metadata_path = version_dir / "metadata.yaml"
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        metadata["active"] = True
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        # Обновление предсказателя
        predictor = ModelPredictor()
        predictor.load_model(str(version_dir))
        
        api_logger.info(f"Model activated: {model_name} v{request.version}")
        
        return {"message": f"Model {model_name} v{request.version} activated"}
        
    except Exception as e:
        api_logger.error(f"Failed to activate model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{model_name}/{version}")
@require_auth()
@require_role(["admin"])
async def delete_model_version(model_name: str, version: str):
    """Удаление версии модели"""
    try:
        version_dir = Path("models") / model_name / version
        
        if not version_dir.exists():
            raise HTTPException(status_code=404, detail="Model version not found")
        
        # Проверка, не является ли эта версия активной
        metadata_path = version_dir / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            if metadata.get("active", False):
                raise HTTPException(
                    status_code=400, 
                    detail="Cannot delete active model version"
                )
        
        # Удаление директории
        shutil.rmtree(version_dir)
        
        api_logger.warning(f"Model version deleted: {model_name} v{version}")
        
        return {"message": f"Model {model_name} v{version} deleted"}
        
    except Exception as e:
        api_logger.error(f"Failed to delete model version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_name}/convert")
@require_auth()
@require_role(["admin", "ml_engineer"])
async def convert_model(
    model_name: str,
    source_framework: str = Form(...),
    target_framework: str = Form("onnx"),
    file: UploadFile = File(...)
):
    """Конвертация модели между фреймворками"""
    try:
        # Сохранение загруженного файла
        upload_dir = Path("uploads") / model_name
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Конвертация
        converter = ModelConverter()
        
        if source_framework == "pytorch" and target_framework == "onnx":
            # Загрузка PyTorch модели
            import torch
            model = torch.load(file_path)
            
            # Конвертация в ONNX
            onnx_path = converter.convert_to_onnx(model, input_size=20)
            
            # Регистрация конвертированной модели
            model_dir = Path("models") / model_name / "converted"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.move(onnx_path, model_dir / "model.onnx")
            
            api_logger.info(f"Model converted: {file.filename} -> ONNX")
            
            return {
                "message": "Model converted successfully",
                "converted_path": str(model_dir / "model.onnx")
            }
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Conversion from {source_framework} to {target_framework} not supported"
            )
        
    except Exception as e:
        api_logger.error(f"Model conversion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}/performance")
@require_auth()
@require_permission(["models:read"])
async def get_model_performance(
    model_name: str,
    version: Optional[str] = None,
    days: int = 30
):
    """Получение метрик производительности модели"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        # Запрос метрик из БД
        query = """
        SELECT 
            model_version,
            AVG(processing_time_ms) as avg_latency,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms) as p95_latency,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY processing_time_ms) as p99_latency,
            COUNT(*) as total_predictions,
            AVG((prediction_result->>'probability')::float) as avg_score,
            STDDEV((prediction_result->>'probability')::float) as score_stddev
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '%s days'
        """
        
        params = [days]
        
        if version:
            query += " AND model_version = %s"
            params.append(version)
        else:
            query += " AND model_version LIKE %s"
            params.append(f"{model_name}%")
        
        query += " GROUP BY model_version"
        
        performance = db_pool.execute_query(query, tuple(params))
        
        return {
            "model_name": model_name,
            "version": version or "all",
            "period_days": days,
            "performance_metrics": performance
        }
        
    except Exception as e:
        api_logger.error(f"Failed to get model performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))