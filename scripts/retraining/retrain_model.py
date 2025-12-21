"""
Переобучение модели кредитного скоринга
Этап 7: Автоматическое переобучение модели
"""
import pandas as pd
import numpy as np
import joblib
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.model = None
        self.metrics = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Установка значений по умолчанию
        defaults = {
            'data': {
                'train_path': 'data/processed/train.csv',
                'test_path': 'data/processed/test.csv',
                'validation_split': 0.2,
                'target_column': 'default'
            },
            'model': {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'output_dir': 'models/retrained',
                'backup_current': True
            },
            'evaluation': {
                'metrics': ['accuracy', 'roc_auc', 'f1', 'precision', 'recall'],
                'threshold_improvement': 0.01
            },
            'retraining': {
                'min_samples': 1000,
                'class_balance_threshold': 0.3
            }
        }
        
        # Рекурсивное обновление конфигурации
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v
        
        update_dict(defaults, config)
        return defaults
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Загрузка и подготовка данных"""
        logger.info("Загрузка данных...")
        
        train_path = Path(self.config['data']['train_path'])
        test_path = Path(self.config['data']['test_path'])
        
        if not train_path.exists():
            raise FileNotFoundError(f"Тренировочные данные не найдены: {train_path}")
        
        if not test_path.exists():
            logger.warning(f"Тестовые данные не найдены: {test_path}")
            # Разделим тренировочные данные на train/test
            test_path = None
        
        # Загрузка тренировочных данных
        train_data = pd.read_csv(train_path)
        logger.info(f"Тренировочные данные: {train_data.shape[0]} строк, {train_data.shape[1]} колонок")
        
        # Загрузка тестовых данных или разделение
        if test_path:
            test_data = pd.read_csv(test_path)
            logger.info(f"Тестовые данные: {test_data.shape[0]} строк, {test_data.shape[1]} колонок")
            
            # Разделение тренировочных данных на train/validation
            from sklearn.model_selection import train_test_split
            
            target_col = self.config['data']['target_column']
            X = train_data.drop(columns=[target_col])
            y = train_data[target_col]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config['data']['validation_split'],
                random_state=42,
                stratify=y
            )
            
            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
            
        else:
            # Разделение на train/validation/test
            from sklearn.model_selection import train_test_split
            
            target_col = self.config['data']['target_column']
            X = train_data.drop(columns=[target_col])
            y = train_data[target_col]
            
            # Сначала отделим тестовые данные
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            
            # Затем разделим оставшиеся на train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=self.config['data']['validation_split'],
                random_state=42,
                stratify=y_train_val
            )
            
            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Данные разделены: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def check_data_quality(self, train_data: pd.DataFrame) -> bool:
        """Проверка качества данных перед обучением"""
        logger.info("Проверка качества данных...")
        
        target_col = self.config['data']['target_column']
        
        # Проверка минимального количества образцов
        min_samples = self.config['retraining']['min_samples']
        if len(train_data) < min_samples:
            logger.warning(f"Мало тренировочных данных: {len(train_data)} < {min_samples}")
            return False
        
        # Проверка баланса классов
        class_counts = train_data[target_col].value_counts()
        class_ratio = class_counts.min() / class_counts.max()
        
        threshold = self.config['retraining']['class_balance_threshold']
        if class_ratio < threshold:
            logger.warning(f"Сильный дисбаланс классов: ratio={class_ratio:.3f} < {threshold}")
            # Можно применить балансировку, но пока просто предупреждение
        
        # Проверка пропущенных значений
        missing_percent = (train_data.isnull().sum() / len(train_data)) * 100
        if missing_percent.max() > 5:
            logger.warning(f"Есть колонки с >5% пропущенных значений")
            # Можно применить импутацию
        
        return True
    
    def create_model(self):
        """Создание модели"""
        logger.info("Создание модели...")
        
        model_type = self.config['model']['type']
        params = self.config['model']['params']
        
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**params)
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(**params)
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(**params)
            except ImportError:
                logger.error("XGBoost не установлен. Установите: pip install xgboost")
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier()
        else:
            logger.warning(f"Неизвестный тип модели: {model_type}, используем RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier()
        
        logger.info(f"Модель создана: {type(self.model).__name__}")
    
    def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> bool:
        """Обучение модели"""
        logger.info("Обучение модели...")
        
        target_col = self.config['data']['target_column']
        
        # Подготовка данных
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        
        X_val = val_data.drop(columns=[target_col])
        y_val = val_data[target_col]
        
        # Обучение
        logger.info(f"Обучение на {len(X_train)} образцах...")
        start_time = datetime.now()
        
        self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Обучение завершено за {training_time:.2f} секунд")
        
        # Оценка на валидационных данных
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        self.metrics['validation'] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'f1_score': f1_score(y_val, y_pred),
            'training_time_seconds': training_time,
            'samples': {
                'train': len(X_train),
                'validation': len(X_val)
            }
        }
        
        logger.info(f"Метрики валидации: accuracy={self.metrics['validation']['accuracy']:.4f}, "
                   f"roc_auc={self.metrics['validation']['roc_auc']:.4f}")
        
        return True
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Оценка модели на тестовых данных"""
        logger.info("Оценка модели на тестовых данных...")
        
        target_col = self.config['data']['target_column']
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]
        
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score,
            precision_score, recall_score, confusion_matrix,
            classification_report
        )
        
        # Предсказания
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Вычисление метрик
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'test_samples': len(X_test)
        }
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        self.metrics['test'] = metrics
        
        logger.info(f"Метрики тестирования: accuracy={metrics['accuracy']:.4f}, "
                   f"roc_auc={metrics['roc_auc']:.4f}, f1={metrics['f1_score']:.4f}")
        
        return metrics
    
    def compare_with_current_model(self) -> Tuple[bool, Dict]:
        """Сравнение с текущей моделью"""
        logger.info("Сравнение с текущей моделью...")
        
        current_model_path = Path('models/trained/credit_scoring_model.pkl')
        
        if not current_model_path.exists():
            logger.warning("Текущая модель не найдена, считаем новую модель лучше")
            return True, {'improvement': 'no_current_model'}
        
        try:
            # Загрузка текущей модели
            current_model = joblib.load(current_model_path)
            
            # Загрузка текущих метрик
            metrics_path = Path('models/trained/performance_metrics.json')
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    current_metrics = json.load(f)
            else:
                logger.warning("Метрики текущей модели не найдены")
                return True, {'improvement': 'no_current_metrics'}
            
            # Сравнение метрик
            improvement = {}
            threshold = self.config['evaluation']['threshold_improvement']
            
            for metric in ['accuracy', 'roc_auc', 'f1_score']:
                if metric in self.metrics['test'] and metric in current_metrics:
                    diff = self.metrics['test'][metric] - current_metrics[metric]
                    improvement[metric] = diff
                    
                    logger.info(f"{metric}: текущая={current_metrics[metric]:.4f}, "
                               f"новая={self.metrics['test'][metric]:.4f}, diff={diff:.4f}")
            
            # Проверка, лучше ли новая модель
            is_better = any(diff > threshold for diff in improvement.values())
            
            return is_better, improvement
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении с текущей моделью: {e}")
            return True, {'error': str(e)}
    
    def save_model(self) -> Path:
        """Сохранение модели и метрик"""
        logger.info("Сохранение модели и метрик...")
        
        output_dir = Path(self.config['model']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение модели
        model_filename = f"credit_scoring_model_{timestamp}.pkl"
        model_path = output_dir / model_filename
        joblib.dump(self.model, model_path)
        
        # Сохранение метрик
        metrics_filename = f"performance_metrics_{timestamp}.json"
        metrics_path = output_dir / metrics_filename
        
        # Добавляем дополнительную информацию
        full_metrics = {
            **self.metrics,
            'model_info': {
                'type': self.config['model']['type'],
                'params': self.config['model']['params'],
                'training_date': timestamp,
                'data_sources': {
                    'train': self.config['data']['train_path'],
                    'test': self.config['data']['test_path']
                }
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(full_metrics, f, indent=2)
        
        # Создание симлинков на последнюю модель
        latest_model_path = output_dir / "credit_scoring_model_latest.pkl"
        latest_metrics_path = output_dir / "performance_metrics_latest.json"
        
        if latest_model_path.exists():
            latest_model_path.unlink()
        latest_model_path.symlink_to(model_filename)
        
        if latest_metrics_path.exists():
            latest_metrics_path.unlink()
        latest_metrics_path.symlink_to(metrics_filename)
        
        logger.info(f"Модель сохранена: {model_path}")
        logger.info(f"Метрики сохранены: {metrics_path}")
        
        return model_path
    
    def backup_current_model(self):
        """Создание резервной копии текущей модели"""
        if not self.config['model']['backup_current']:
            return
        
        current_model_path = Path('models/trained/credit_scoring_model.pkl')
        if not current_model_path.exists():
            return
        
        backup_dir = Path('models/backup')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"credit_scoring_model_backup_{timestamp}.pkl"
        
        import shutil
        shutil.copy2(current_model_path, backup_path)
        
        logger.info(f"Резервная копия создана: {backup_path}")
    
    def deploy_model(self, model_path: Path):
        """Деплой модели в рабочую директорию"""
        logger.info("Деплой модели...")
        
        target_dir = Path('models/trained')
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_model_path = target_dir / 'credit_scoring_model.pkl'
        target_metrics_path = target_dir / 'performance_metrics.json'
        
        # Копирование модели
        import shutil
        shutil.copy2(model_path, target_model_path)
        
        # Копирование метрик
        metrics_path = model_path.parent / f"performance_metrics_{model_path.stem.split('_')[-1]}.json"
        if metrics_path.exists():
            shutil.copy2(metrics_path, target_metrics_path)
        
        logger.info(f"Модель развернута: {target_model_path}")
    
    def run(self, deploy: bool = True) -> bool:
        """Запуск процесса переобучения"""
        logger.info("Запуск процесса переобучения модели...")
        
        try:
            # 1. Загрузка данных
            train_data, val_data, test_data = self.load_data()
            
            # 2. Проверка качества данных
            if not self.check_data_quality(train_data):
                logger.warning("Проблемы с качеством данных, но продолжаем...")
            
            # 3. Создание модели
            self.create_model()
            
            # 4. Обучение модели
            if not self.train_model(train_data, val_data):
                logger.error("Не удалось обучить модель")
                return False
            
            # 5. Оценка на тестовых данных
            test_metrics = self.evaluate_model(test_data)
            
            # 6. Сравнение с текущей моделью
            is_better, improvement = self.compare_with_current_model()
            
            if not is_better:
                logger.warning("Новая модель не лучше текущей")
                logger.info(f"Улучшение метрик: {improvement}")
                
                threshold = self.config['evaluation']['threshold_improvement']
                if all(diff < threshold for diff in improvement.values() if isinstance(diff, (int, float))):
                    logger.info("Модель не улучшила метрики достаточно, прерываем деплой")
                    deploy = False
            
            # 7. Сохранение модели
            model_path = self.save_model()
            
            # 8. Деплой если нужно
            if deploy:
                self.backup_current_model()
                self.deploy_model(model_path)
                logger.info("Модель успешно переобучена и развернута")
            else:
                logger.info("Модель переобучена, но не развернута (недостаточное улучшение)")
            
            # 9. Генерация отчета
            self.generate_report(model_path, is_better, improvement)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при переобучении модели: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_report(self, model_path: Path, is_better: bool, improvement: Dict):
        """Генерация отчета о переобучении"""
        logger.info("Генерация отчета...")
        
        report_dir = Path('reports/retraining')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'retraining_report': {
                'timestamp': timestamp,
                'model_path': str(model_path),
                'is_better_than_current': is_better,
                'improvement_metrics': improvement,
                'new_model_metrics': self.metrics,
                'config': self.config
            }
        }
        
        report_path = report_dir / f"retraining_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Генерация HTML отчета
        html_path = report_dir / f"retraining_report_{timestamp}.html"
        self.generate_html_report(html_path, report)
        
        logger.info(f"Отчет сохранен: {report_path}")
    
    def generate_html_report(self, output_path: Path, report_data: Dict):
        """Генерация HTML отчета"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Отчет переобучения модели - {report_data['retraining_report']['timestamp']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 5px; }}
                .better {{ color: green; font-weight: bold; }}
                .worse {{ color: red; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Отчет переобучения модели</h1>
                <p>Дата: {report_data['retraining_report']['timestamp']}</p>
                <p>Модель: {report_data['retraining_report']['model_path']}</p>
                <p>Статус: <span class="{'better' if report_data['retraining_report']['is_better_than_current'] else 'worse'}">
                    {'ЛУЧШЕ текущей' if report_data['retraining_report']['is_better_than_current'] else 'ХУЖЕ текущей'}
                </span></p>
            </div>
            
            <div class="section">
                <h2>Метрики новой модели</h2>
                <div class="metrics">
        """
        
        # Добавление метрик
        metrics = report_data['retraining_report']['new_model_metrics']
        if 'test' in metrics:
            for name, value in metrics['test'].items():
                if isinstance(value, (int, float)):
                    html_content += f'<div class="metric"><strong>{name}:</strong> {value:.4f}</div>'
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Сравнение с текущей моделью</h2>
                <table>
                    <tr><th>Метрика</th><th>Улучшение</th></tr>
        """
        
        # Добавление сравнения
        improvement = report_data['retraining_report']['improvement_metrics']
        for metric, value in improvement.items():
            if isinstance(value, (int, float)):
                html_content += f'<tr><td>{metric}</td><td class="{"better" if value > 0 else "worse"}">{value:+.4f}</td></tr>'
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Информация о данных</h2>
                <p><strong>Тренировочные данные:</strong> 
        """
        
        if 'validation' in metrics and 'samples' in metrics['validation']:
            html_content += f"{metrics['validation']['samples'].get('train', 'N/A')} образцов</p>"
        
        html_content += """
                <p><strong>Тестовые данные:</strong> 
        """
        
        if 'test' in metrics:
            html_content += f"{metrics['test'].get('test_samples', 'N/A')} образцов</p>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Рекомендации</h2>
        """
        
        if report_data['retraining_report']['is_better_than_current']:
            html_content += "<p>✅ Новая модель показывает улучшение метрик. Рекомендуется деплой.</p>"
        else:
            html_content += "<p>⚠️ Новая модель не показывает значительного улучшения. Рассмотрите возможность отката.</p>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Переобучение модели кредитного скоринга')
    parser.add_argument('--config', default='configs/retraining_config.yaml',
                       help='Файл конфигурации')
    parser.add_argument('--no-deploy', action='store_true',
                       help='Не развертывать модель после обучения')
    parser.add_argument('--force', action='store_true',
                       help='Принудительное развертывание даже без улучшения')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ПЕРЕОБУЧЕНИЕ МОДЕЛИ КРЕДИТНОГО СКОРИНГА")
    logger.info("=" * 60)
    
    # Инициализация ретренера
    retrainer = ModelRetrainer(args.config)
    
    # Запуск переобучения
    deploy = not args.no_deploy
    if args.force:
        deploy = True
    
    success = retrainer.run(deploy=deploy)
    
    if success:
        logger.info("Переобучение завершено успешно!")
        return 0
    else:
        logger.error("Переобучение завершилось с ошибкой")
        return 1

if __name__ == "__main__":
    exit(main())