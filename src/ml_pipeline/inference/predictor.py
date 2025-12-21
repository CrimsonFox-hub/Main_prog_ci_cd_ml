"""
Прогнозирование с использованием обученных моделей.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Класс для прогнозирования с использованием обученных моделей"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.feature_names = None
        self.loaded_at = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Загрузка модели из файла"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_path = model_path
            self.loaded_at = datetime.now()
            logger.info(f"Модель загружена из {model_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def predict(self, features: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Предсказание для одного или нескольких образцов"""
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        features_array = np.array(features)
        
        # Проверка размерности
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        try:
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(features_array)[:, 1]
            elif hasattr(self.model, 'predict'):
                predictions = self.model.predict(features_array)
            else:
                raise AttributeError("Модель не поддерживает predict или predict_proba")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise
    
    def predict_batch(self, 
                     features_batch: List[List[float]], 
                     threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Пакетное предсказание с порогом"""
        predictions = self.predict(features_batch)
        results = []
        
        for i, pred in enumerate(predictions):
            results.append({
                "sample_id": i,
                "prediction": float(pred),
                "prediction_class": int(pred > threshold),
                "threshold": threshold
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о загруженной модели"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_path": self.model_path,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "model_type": type(self.model).__name__,
            "has_predict_proba": hasattr(self.model, 'predict_proba'),
            "has_predict": hasattr(self.model, 'predict')
        }
        
        # Дополнительная информация о модели
        if hasattr(self.model, 'get_params'):
            try:
                info["params"] = self.model.get_params()
            except:
                info["params"] = "unavailable"
        
        return info


# Глобальный экземпляр для использования в API
_global_predictor = None

def get_predictor(model_path: Optional[str] = None) -> ModelPredictor:
    """Получение глобального экземпляра предсказателя"""
    global _global_predictor
    
    if _global_predictor is None and model_path:
        _global_predictor = ModelPredictor(model_path)
    elif _global_predictor is None:
        # Попробуем найти модель по умолчанию
        default_paths = [
            "models/trained/model.pkl",
            "models/trained/credit_scoring_model.pkl",
            "model.pkl"
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                _global_predictor = ModelPredictor(path)
                break
        
        if _global_predictor is None:
            logger.warning("Модель не найдена, создан пустой предсказатель")
            _global_predictor = ModelPredictor()
    
    return _global_predictor