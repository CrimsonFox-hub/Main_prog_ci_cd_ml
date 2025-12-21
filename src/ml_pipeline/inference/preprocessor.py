"""
Препроцессинг признаков для модели
"""
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from src.utils.logger import model_logger
from src.utils.config_loader import get_config

class FeaturePreprocessor:
    """Класс для препроцессинга признаков"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config('model', config_path)
        
        # Загрузка препроцессоров
        self.scaler = None
        self.encoder = None
        self.imputer = None
        
        self._load_preprocessors()
        
        # Конфигурация признаков
        self.numerical_features = self.config.get('numerical_features', [])
        self.categorical_features = self.config.get('categorical_features', [])
        self.feature_ranges = self.config.get('feature_ranges', {})
    
    def _load_preprocessors(self):
        """Загрузка сохраненных препроцессоров"""
        try:
            preprocessors_path = Path(self.config.get('preprocessors_path', 'models/preprocessors.joblib'))
            
            if preprocessors_path.exists():
                preprocessors = joblib.load(preprocessors_path)
                
                self.scaler = preprocessors.get('scaler')
                self.encoder = preprocessors.get('encoder')
                self.imputer = preprocessors.get('imputer')
                
                model_logger.info("Preprocessors loaded successfully")
            else:
                model_logger.warning("Preprocessors file not found")
                
        except Exception as e:
            model_logger.error(f"Failed to load preprocessors: {e}", exc_info=True)
    
    def validate_features(self, features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Валидация входящих признаков"""
        errors = []
        
        # Проверка обязательных признаков
        required_features = self.config.get('required_features', [])
        for feature in required_features:
            if feature not in features:
                errors.append(f"Missing required feature: {feature}")
        
        # Проверка типов и диапазонов
        for feature_name, value in features.items():
            # Проверка типа
            if feature_name in self.numerical_features:
                if not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except:
                        errors.append(f"Feature {feature_name} must be numerical")
                
                # Проверка диапазона
                if feature_name in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[feature_name]
                    if value < min_val or value > max_val:
                        errors.append(f"Feature {feature_name} out of range [{min_val}, {max_val}]")
            
            # Проверка категориальных признаков
            elif feature_name in self.categorical_features:
                allowed_values = self.config.get('allowed_values', {}).get(feature_name, [])
                if allowed_values and value not in allowed_values:
                    errors.append(f"Invalid value for {feature_name}. Allowed: {allowed_values}")
        
        return len(errors) == 0, errors
    
    def preprocess_single(self, features: Dict[str, Any]) -> np.ndarray:
        """Препроцессинг одного наблюдения"""
        # Валидация
        is_valid, errors = self.validate_features(features)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {errors}")
        
        # Сбор признаков в правильном порядке
        processed = []
        
        # Числовые признаки
        for feature in self.numerical_features:
            value = features.get(feature, np.nan)
            
            # Обработка пропущенных значений
            if pd.isna(value):
                if self.imputer:
                    # Будет обработано позже
                    processed.append(np.nan)
                else:
                    processed.append(0.0)  # Дефолтное значение
            else:
                processed.append(float(value))
        
        # Категориальные признаки
        for feature in self.categorical_features:
            value = features.get(feature, 'missing')
            
            if self.encoder and feature in self.encoder.feature_names_in_:
                try:
                    # One-hot encoding
                    encoded = self.encoder.transform([[value]]).toarray()[0]
                    processed.extend(encoded.tolist())
                except:
                    # Если значение не было в тренировочных данных
                    processed.extend([0] * len(self.encoder.categories_[0]))
            else:
                # Simple label encoding для новых признаков
                processed.append(hash(value) % 1000)
        
        # Конвертация в numpy array
        features_array = np.array([processed], dtype=np.float32)
        
        # Импутация пропущенных значений
        if self.imputer and np.any(np.isnan(features_array)):
            features_array = self.imputer.transform(features_array)
        
        # Масштабирование
        if self.scaler:
            features_array = self.scaler.transform(features_array)
        
        return features_array
    
    def preprocess_batch(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """Препроцессинг батча наблюдений"""
        processed_batch = []
        
        for features in features_list:
            try:
                processed = self.preprocess_single(features)
                processed_batch.append(processed[0])  # Удаляем batch dimension
            except Exception as e:
                model_logger.error(f"Failed to preprocess features: {e}")
                # Добавление нулевого вектора для неудачных наблюдений
                if processed_batch:
                    processed_batch.append(np.zeros_like(processed_batch[0]))
                else:
                    # Если это первый элемент, создаем дефолтный вектор
                    num_features = len(self.numerical_features)
                    cat_features = len(self.categorical_features) * 10  # Примерная длина
                    processed_batch.append(np.zeros(num_features + cat_features))
        
        return np.array(processed_batch, dtype=np.float32)
    
    def inverse_transform(self, scaled_features: np.ndarray) -> Dict[str, Any]:
        """Обратное преобразование признаков"""
        if not self.scaler:
            raise ValueError("Scaler not loaded")
        
        # Обратное масштабирование
        original_features = self.scaler.inverse_transform(scaled_features)
        
        # Формирование словаря признаков
        features_dict = {}
        
        for i, feature_name in enumerate(self.numerical_features):
            if i < len(original_features[0]):
                features_dict[feature_name] = float(original_features[0][i])
        
        return features_dict
    
    def get_feature_names(self) -> List[str]:
        """Получение имен всех признаков после препроцессинга"""
        feature_names = []
        
        # Числовые признаки
        feature_names.extend(self.numerical_features)
        
        # Категориальные признаки (one-hot encoded)
        if self.encoder:
            for i, feature in enumerate(self.categorical_features):
                if feature in self.encoder.feature_names_in_:
                    categories = self.encoder.categories_[i]
                    for category in categories:
                        feature_names.append(f"{feature}_{category}")
        
        return feature_names
    
    def save_preprocessors(self, path: str):
        """Сохранение препроцессоров"""
        preprocessors = {
            'scaler': self.scaler,
            'encoder': self.encoder,
            'imputer': self.imputer
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessors, path)
        
        model_logger.info(f"Preprocessors saved to: {path}")
    
    def load_preprocessors(self, path: str):
        """Загрузка препроцессоров"""
        try:
            preprocessors = joblib.load(path)
            
            self.scaler = preprocessors.get('scaler')
            self.encoder = preprocessors.get('encoder')
            self.imputer = preprocessors.get('imputer')
            
            model_logger.info(f"Preprocessors loaded from: {path}")
            
        except Exception as e:
            model_logger.error(f"Failed to load preprocessors: {e}", exc_info=True)
            raise

class FeaturePipeline:
    """Пайплайн для обработки признаков"""
    
    def __init__(self):
        self.steps = []
        self.preprocessor = FeaturePreprocessor()
    
    def add_step(self, name: str, function):
        """Добавление шага в пайплайн"""
        self.steps.append({'name': name, 'function': function})
    
    def process(self, features: Dict[str, Any]) -> np.ndarray:
        """Обработка признаков через все шаги пайплайна"""
        result = features
        
        for step in self.steps:
            try:
                result = step['function'](result)
                model_logger.debug(f"Pipeline step '{step['name']}' completed")
            except Exception as e:
                model_logger.error(f"Pipeline step '{step['name']}' failed: {e}")
                raise
        
        # Финальный препроцессинг
        return self.preprocessor.preprocess_single(result)