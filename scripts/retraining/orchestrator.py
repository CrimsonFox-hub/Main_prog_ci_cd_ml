"""
Оркестратор переобучения модели
Этап 7: Автоматическое переобучение с триггерами
"""
import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetrainingOrchestrator:
    """Оркестратор автоматического переобучения моделей"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.results = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Настройки по умолчанию
        defaults = {
            'retraining': {
                'enabled': True,
                'schedule': 'weekly',
                'trigger_on_drift': True,
                'trigger_on_performance': True,
                'min_data_samples': 1000,
                'validation_threshold': 0.02
            },
            'data': {
                'source_path': 'data/processed/train.csv',
                'test_path': 'data/processed/test.csv',
                'target_column': 'default'
            },
            'model': {
                'output_dir': 'models/retrained',
                'backup_dir': 'models/backup',
                'current_model_path': 'models/trained/credit_scoring_model.pkl'
            },
            'monitoring': {
                'drift_threshold': 0.3,
                'performance_threshold': 0.05
            }
        }
        
        # Объединяем конфигурации
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            else:
                config[key].update(value)
        
        return config
    
    def check_triggers(self) -> Dict[str, bool]:
        """Проверка триггеров для переобучения"""
        logger.info("Проверка триггеров для переобучения...")
        
        triggers = {
            'scheduled': False,
            'data_drift': False,
            'performance_decay': False,
            'new_data': False
        }
        
        # Триггер по расписанию
        if self.config['retraining']['schedule'] == 'daily':
            # Проверяем, прошли ли сутки с последнего переобучения
            last_retraining = self.get_last_retraining_time()
            if last_retraining is None or (datetime.now() - last_retraining).days >= 1:
                triggers['scheduled'] = True
        
        # Триггер по дрифту данных (используем скрипт check_drift)
        if self.config['retraining']['trigger_on_drift']:
            drift_detected = self.check_data_drift()
            triggers['data_drift'] = drift_detected
        
        # Триггер по ухудшению производительности
        if self.config['retraining']['trigger_on_performance']:
            performance_decay = self.check_performance_decay()
            triggers['performance_decay'] = performance_decay
        
        # Триггер по новым данным
        new_data_available = self.check_new_data()
        triggers['new_data'] = new_data_available
        
        logger.info(f"Триггеры: {triggers}")
        return triggers
    
    def get_last_retraining_time(self) -> Optional[datetime]:
        """Получение времени последнего переобучения"""
        backup_dir = Path(self.config['model']['backup_dir'])
        
        if not backup_dir.exists():
            return None
        
        # Ищем последний файл модели
        model_files = list(backup_dir.glob('*.pkl'))
        if not model_files:
            return None
        
        latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
        return datetime.fromtimestamp(latest_file.stat().st_mtime)
    
    def check_data_drift(self) -> bool:
        """Проверка дрифта данных"""
        try:
            # Импортируем здесь, чтобы не зависеть от evidently при установке
            from scripts.monitoring.check_drift import DriftMonitor
            
            monitor = DriftMonitor()
            
            # Загрузка данных
            reference_path = self.config['data']['source_path']
            current_path = 'data/processed/latest_production_data.csv'
            
            if not Path(current_path).exists():
                logger.warning(f"Текущие данные не найдены: {current_path}")
                return False
            
            if not monitor.load_reference_data(reference_path):
                return False
            
            if not monitor.load_current_data(current_path):
                return False
            
            # Проверка дрифта
            result = monitor.check_data_drift()
            
            if result and result.get('should_retrain', False):
                logger.info("Обнаружен дрифт данных, требуется переобучение")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка при проверке дрифта данных: {e}")
            return False
    
    def check_performance_decay(self) -> bool:
        """Проверка ухудшения производительности модели"""
        try:
            # Получаем текущие метрики производительности
            current_metrics = self.get_current_performance_metrics()
            
            if not current_metrics:
                return False
            
            # Получаем эталонные метрики
            reference_metrics = self.get_reference_performance_metrics()
            
            if not reference_metrics:
                return False
            
            # Сравниваем метрики
            threshold = self.config['monitoring']['performance_threshold']
            
            for metric_name in ['accuracy', 'roc_auc', 'f1_score']:
                if metric_name in current_metrics and metric_name in reference_metrics:
                    decay = reference_metrics[metric_name] - current_metrics[metric_name]
                    
                    if decay > threshold:
                        logger.info(f"Ухудшение {metric_name}: {decay:.3f} > {threshold}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка при проверке производительности: {e}")
            return False
    
    def get_current_performance_metrics(self) -> Dict:
        """Получение текущих метрик производительности"""
        metrics_path = Path('models/trained/performance_metrics.json')
        
        if not metrics_path.exists():
            return {}
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def get_reference_performance_metrics(self) -> Dict:
        """Получение эталонных метрик производительности"""
        reference_path = Path('models/trained/reference_metrics.json')
        
        if not reference_path.exists():
            # Если нет эталонных метрик, используем текущие
            return self.get_current_performance_metrics()
        
        with open(reference_path, 'r') as f:
            return json.load(f)
    
    def check_new_data(self) -> bool:
        """Проверка наличия новых данных"""
        source_path = Path(self.config['data']['source_path'])
        new_data_path = Path('data/raw/new_credit_data.csv')
        
        if not new_data_path.exists():
            return False
        
        # Проверяем размер новых данных
        new_data_size = new_data_path.stat().st_size
        source_data_size = source_path.stat().st_size
        
        # Если новые данные больше определенного процента от текущих
        min_samples = self.config['retraining']['min_data_samples']
        
        try:
            import pandas as pd
            new_data = pd.read_csv(new_data_path)
            
            if len(new_data) >= min_samples:
                logger.info(f"Новые данные доступны: {len(new_data)} образцов")
                return True
            
        except Exception as e:
            logger.error(f"Ошибка при проверке новых данных: {e}")
        
        return False
    
    def prepare_data(self) -> bool:
        """Подготовка данных для переобучения"""
        logger.info("Подготовка данных для переобучения...")
        
        try:
            # Объединяем старые и новые данные
            import pandas as pd
            from sklearn.model_selection import train_test_split
            
            # Загружаем текущие данные
            current_data_path = Path(self.config['data']['source_path'])
            if not current_data_path.exists():
                logger.error(f"Текущие данные не найдены: {current_data_path}")
                return False
            
            current_data = pd.read_csv(current_data_path)
            
            # Загружаем новые данные если есть
            new_data_path = Path('data/raw/new_credit_data.csv')
            if new_data_path.exists():
                new_data = pd.read_csv(new_data_path)
                combined_data = pd.concat([current_data, new_data], ignore_index=True)
                
                # Сохраняем объединенные данные
                combined_path = Path('data/processed/combined_train.csv')
                combined_data.to_csv(combined_path, index=False)
                
                logger.info(f"Объединенные данные: {len(combined_data)} образцов")
            else:
                combined_path = current_data_path
            
            # Разделяем на train/validation
            data = pd.read_csv(combined_path)
            target_col = self.config['data']['target_column']
            
            if target_col not in data.columns:
                logger.error(f"Целевая переменная не найдена: {target_col}")
                return False
            
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Сохраняем разделенные данные
            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
            
            train_path = Path('data/processed/retrain_train.csv')
            val_path = Path('data/processed/retrain_val.csv')
            
            train_data.to_csv(train_path, index=False)
            val_data.to_csv(val_path, index=False)
            
            logger.info(f"Данные подготовлены: train={len(train_data)}, val={len(val_data)}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            return False
    
    def retrain_model(self) -> Optional[str]:
        """Переобучение модели"""
        logger.info("Запуск переобучения модели...")
        
        try:
            # Импортируем здесь, чтобы не зависеть от ML библиотек в основном коде
            import pandas as pd
            import joblib
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
            
            # Загрузка данных
            train_path = Path('data/processed/retrain_train.csv')
            val_path = Path('data/processed/retrain_val.csv')
            
            if not train_path.exists() or not val_path.exists():
                logger.error("Подготовленные данные не найдены")
                return None
            
            train_data = pd.read_csv(train_path)
            val_data = pd.read_csv(val_path)
            
            target_col = self.config['data']['target_column']
            
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col]
            X_val = val_data.drop(columns=[target_col])
            y_val = val_data[target_col]
            
            # Обучение модели
            logger.info(f"Обучение модели на {len(X_train)} образцах...")
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Оценка модели
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba),
                'f1_score': f1_score(y_val, y_pred)
            }
            
            logger.info(f"Метрики валидации: {metrics}")
            
            # Сохранение модели
            output_dir = Path(self.config['model']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = output_dir / f"model_{timestamp}.pkl"
            
            joblib.dump(model, model_path)
            
            # Сохранение метрик
            metrics_path = output_dir / f"metrics_{timestamp}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Модель сохранена: {model_path}")
            logger.info(f"Метрики сохранены: {metrics_path}")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Ошибка при переобучении модели: {e}")
            return None
    
    def evaluate_model(self, model_path: str) -> Tuple[bool, Dict]:
        """Оценка переобученной модели"""
        logger.info(f"Оценка модели: {model_path}")
        
        try:
            import pandas as pd
            import joblib
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
            
            # Загрузка тестовых данных
            test_path = Path(self.config['data']['test_path'])
            if not test_path.exists():
                logger.warning("Тестовые данные не найдены, используем валидационные")
                test_path = Path('data/processed/retrain_val.csv')
            
            test_data = pd.read_csv(test_path)
            target_col = self.config['data']['target_column']
            
            X_test = test_data.drop(columns=[target_col])
            y_test = test_data[target_col]
            
            # Загрузка модели
            model = joblib.load(model_path)
            
            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Метрики
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'f1_score': f1_score(y_test, y_pred),
                'test_samples': len(X_test),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Метрики тестирования: {metrics}")
            
            # Сравнение с текущей моделью
            current_metrics = self.get_current_performance_metrics()
            
            if current_metrics:
                improvement = {}
                for metric in ['accuracy', 'roc_auc', 'f1_score']:
                    if metric in metrics and metric in current_metrics:
                        improvement[metric] = metrics[metric] - current_metrics[metric]
                
                metrics['improvement'] = improvement
                logger.info(f"Улучшение метрик: {improvement}")
                
                # Проверка порога улучшения
                threshold = self.config['retraining']['validation_threshold']
                is_better = any(imp > threshold for imp in improvement.values())
                
                return is_better, metrics
            
            return True, metrics  # Если нет текущих метрик, считаем модель лучше
            
        except Exception as e:
            logger.error(f"Ошибка при оценке модели: {e}")
            return False, {}
    
    def deploy_model(self, model_path: str) -> bool:
        """Деплой новой модели"""
        logger.info(f"Деплой модели: {model_path}")
        
        try:
            # Создаем бэкап текущей модели
            current_model_path = Path(self.config['model']['current_model_path'])
            backup_dir = Path(self.config['model']['backup_dir'])
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            if current_model_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"backup_{timestamp}.pkl"
                
                import shutil
                shutil.copy2(current_model_path, backup_path)
                logger.info(f"Бэкап создан: {backup_path}")
            
            # Копируем новую модель на место текущей
            shutil.copy2(model_path, current_model_path)
            logger.info(f"Модель развернута: {current_model_path}")
            
            # Обновляем эталонные метрики
            metrics_path = Path('models/trained/reference_metrics.json')
            with open(metrics_path, 'w') as f:
                import json
                json.dump(self.results.get('evaluation_metrics', {}), f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при деплое модели: {e}")
            return False
    
    def cleanup(self):
        """Очистка временных файлов"""
        logger.info("Очистка временных файлов...")
        
        try:
            import shutil
            
            temp_files = [
                'data/processed/combined_train.csv',
                'data/processed/retrain_train.csv',
                'data/processed/retrain_val.csv'
            ]
            
            for file_path in temp_files:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    logger.debug(f"Удален: {file_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при очистке: {e}")
    
    def run(self, force_retrain: bool = False) -> bool:
        """Запуск оркестратора переобучения"""
        logger.info("Запуск оркестратора переобучения...")
        
        # Проверка триггеров
        if not force_retrain:
            triggers = self.check_triggers()
            
            # Проверяем, есть ли хотя бы один активный триггер
            should_retrain = any(triggers.values())
            
            if not should_retrain:
                logger.info("Нет активных триггеров для переобучения")
                return False
        
        logger.info("Начало процесса переобучения")
        
        try:
            # 1. Подготовка данных
            if not self.prepare_data():
                logger.error("Не удалось подготовить данные")
                return False
            
            # 2. Переобучение модели
            model_path = self.retrain_model()
            if not model_path:
                logger.error("Не удалось переобучить модель")
                return False
            
            # 3. Оценка модели
            is_better, metrics = self.evaluate_model(model_path)
            self.results['evaluation_metrics'] = metrics
            
            if not is_better:
                logger.warning("Новая модель не улучшает метрики")
                self.cleanup()
                return False
            
            # 4. Деплой модели
            if not self.deploy_model(model_path):
                logger.error("Не удалось развернуть модель")
                return False
            
            # 5. Очистка
            self.cleanup()
            
            logger.info("Процесс переобучения завершен успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка в процессе переобучения: {e}")
            self.cleanup()
            return False

def main():
    parser = argparse.ArgumentParser(description='Оркестратор автоматического переобучения моделей')
    parser.add_argument('--config', default='configs/retraining_config.yaml',
                       help='Файл конфигурации')
    parser.add_argument('--force', action='store_true',
                       help='Принудительное переобучение без проверки триггеров')
    parser.add_argument('--dry-run', action='store_true',
                       help='Проверка триггеров без реального переобучения')
    
    args = parser.parse_args()
    
    # Инициализация оркестратора
    orchestrator = RetrainingOrchestrator(args.config)
    
    if args.dry_run:
        # Только проверка триггеров
        triggers = orchestrator.check_triggers()
        print(f"Триггеры переобучения: {triggers}")
        
        if any(triggers.values()):
            print("Рекомендация: запустить переобучение")
            return 1
        else:
            print("Переобучение не требуется")
            return 0
    else:
        # Запуск полного процесса
        success = orchestrator.run(args.force)
        
        if success:
            print("Переобучение завершено успешно")
            return 0
        else:
            print("Переобучение не выполнено")
            return 1

if __name__ == "__main__":
    sys.exit(main())