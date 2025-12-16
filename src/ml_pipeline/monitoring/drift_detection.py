"""
Мониторинг дрифта данных и концепта
Этап 6: Мониторинг дрифта и управление моделями
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path

# Evidently AI для мониторинга дрифта
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
    ColumnSummaryMetric,
    ColumnCorrelationsMetric,
    ColumnValueRangeMetric
)
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
    RegressionPreset,
    ClassificationPreset
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
    TestNumberOfDriftedColumns,
    TestColumnValueMin,
    TestColumnValueMax,
    TestColumnValueMean,
    TestColumnValueStd,
    TestShareOfMissingValues,
    TestNumberOfMissingValues
)

# MLflow для отслеживания
import mlflow

# Prometheus для экспорта метрик
from prometheus_client import Counter, Gauge, Histogram, start_http_server

class DriftMonitor:
    """Класс для мониторинга дрифта данных и концепта"""
    
    def __init__(self, config_path: str = 'configs/monitoring_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Загрузка референсных данных
        self.reference_data = self._load_reference_data()
        
        # Определение колонок
        self.column_mapping = ColumnMapping(
            target=self.config['data']['target_column'],
            prediction='prediction',
            numerical_features=self.config['data']['numerical_features'],
            categorical_features=self.config['data']['categorical_features'],
            datetime_feature=self.config['data'].get('datetime_feature'),
            task='classification'
        )
        
        # Инициализация Prometheus метрик
        self._init_prometheus_metrics()
        
        # Настройка MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        
    def _load_reference_data(self) -> pd.DataFrame:
        """Загрузка референсных данных (тренировочный набор)"""
        ref_path = Path(self.config['data']['reference_path'])
        
        if ref_path.exists():
            data = pd.read_csv(ref_path)
            self.logger.info(f"Loaded reference data with shape: {data.shape}")
            return data
        else:
            raise FileNotFoundError(f"Reference data not found at: {ref_path}")
    
    def _init_prometheus_metrics(self):
        """Инициализация Prometheus метрик"""
        # Метрики дрифта данных
        self.data_drift_score = Gauge(
            'data_drift_score',
            'Overall data drift score',
            ['model_version']
        )
        
        self.column_drift_detected = Gauge(
            'column_drift_detected',
            'Number of columns with detected drift',
            ['model_version', 'column_type']
        )
        
        # Метрики дрифта концепта
        self.concept_drift_score = Gauge(
            'concept_drift_score',
            'Concept drift score',
            ['model_version']
        )
        
        self.model_performance_decay = Gauge(
            'model_performance_decay',
            'Model performance decay percentage',
            ['model_version', 'metric']
        )
        
        # Гистограммы для распределений
        self.feature_distribution_drift = Histogram(
            'feature_distribution_drift',
            'Distribution drift for features',
            ['feature', 'model_version'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        )
        
        # Счетчики событий
        self.drift_alerts_total = Counter(
            'drift_alerts_total',
            'Total number of drift alerts',
            ['alert_level', 'model_version']
        )
        
    def collect_current_data(self, hours: int = 24) -> pd.DataFrame:
        """Сбор текущих данных за указанный период"""
        from src.utils.database import DatabaseConnector
        
        db = DatabaseConnector()
        
        query = f"""
        SELECT * FROM credit_scoring_predictions
        WHERE prediction_time >= NOW() - INTERVAL '{hours} hours'
        AND prediction_time < NOW()
        ORDER BY prediction_time DESC
        """
        
        current_data = db.execute_query(query)
        self.logger.info(f"Collected {len(current_data)} current samples")
        
        return current_data
    
    def calculate_data_drift(self, current_data: pd.DataFrame) -> Dict:
        """Расчет дрифта данных"""
        self.logger.info("Calculating data drift...")
        
        # Создание отчета о дрифте данных
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Извлечение результатов
        report_results = data_drift_report.as_dict()
        
        # Анализ дрифта по колонкам
        column_drift_results = {}
        drifted_columns = []
        
        for column in self.column_mapping.numerical_features + self.column_mapping.categorical_features:
            column_report = Report(metrics=[
                ColumnDriftMetric(column_name=column),
                ColumnSummaryMetric(column_name=column),
            ])
            
            column_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            column_result = column_report.as_dict()
            drift_detected = column_result['metrics'][0]['result']['drift_detected']
            drift_score = column_result['metrics'][0]['result']['drift_score']
            
            column_drift_results[column] = {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'test_name': column_result['metrics'][0]['result']['test_name'],
                'statistical_test': column_result['metrics'][0]['result']['statistical_test']
            }
            
            if drift_detected:
                drifted_columns.append(column)
                
                # Экспорт метрик в Prometheus
                self.feature_distribution_drift.labels(
                    feature=column,
                    model_version=self.config['model']['version']
                ).observe(drift_score)
        
        # Общий дрифт датасета
        dataset_drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric()
        ])
        
        dataset_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        dataset_result = dataset_drift_report.as_dict()
        dataset_drift_score = dataset_result['metrics'][0]['result']['dataset_drift']
        share_of_drifted_columns = dataset_result['metrics'][0]['result']['share_of_drifted_columns']
        
        # Экспорт метрик
        self.data_drift_score.labels(
            model_version=self.config['model']['version']
        ).set(dataset_drift_score)
        
        self.column_drift_detected.labels(
            model_version=self.config['model']['version'],
            column_type='numerical'
        ).set(len([c for c in drifted_columns if c in self.column_mapping.numerical_features]))
        
        self.column_drift_detected.labels(
            model_version=self.config['model']['version'],
            column_type='categorical'
        ).set(len([c for c in drifted_columns if c in self.column_mapping.categorical_features]))
        
        # Сохранение отчета
        report_path = Path(f"monitoring/reports/data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        data_drift_report.save_html(str(report_path))
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift_score': dataset_drift_score,
            'share_of_drifted_columns': share_of_drifted_columns,
            'drifted_columns': drifted_columns,
            'column_drift_details': column_drift_results,
            'total_samples_current': len(current_data),
            'total_samples_reference': len(self.reference_data),
            'report_path': str(report_path),
            'drift_detected': dataset_drift_score > self.config['thresholds']['data_drift']
        }
        
        # Логирование в MLflow
        with mlflow.start_run(run_name=f"drift_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_metrics({
                'data_drift_score': dataset_drift_score,
                'share_of_drifted_columns': share_of_drifted_columns,
                'num_drifted_columns': len(drifted_columns)
            })
            
            mlflow.log_artifact(str(report_path))
            mlflow.log_dict(results, 'drift_results.json')
        
        return results
    
    def calculate_concept_drift(self, current_data: pd.DataFrame) -> Dict:
        """Расчет дрифта концепта (если есть истинные метки)"""
        self.logger.info("Calculating concept drift...")
        
        if self.column_mapping.target not in current_data.columns:
            self.logger.warning("Target column not found in current data. Skipping concept drift calculation.")
            return {}
        
        # Отчет о дрифте целевой переменной
        target_drift_report = Report(metrics=[
            TargetDriftPreset()
        ])
        
        target_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        target_results = target_drift_report.as_dict()
        
        # Извлечение метрик производительности
        performance_report = Report(metrics=[
            ClassificationPreset()
        ])
        
        performance_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        performance_results = performance_report.as_dict()
        
        # Сравнение метрик производительности
        reference_performance = self._get_reference_performance()
        current_performance = performance_results['metrics'][0]['result']['metrics']
        
        performance_decay = {}
        for metric_name, ref_value in reference_performance.items():
            if metric_name in current_performance:
                curr_value = current_performance[metric_name]
                decay = ((ref_value - curr_value) / ref_value) * 100 if ref_value > 0 else 0
                performance_decay[metric_name] = {
                    'reference': ref_value,
                    'current': curr_value,
                    'decay_percent': decay,
                    'significant_decay': abs(decay) > self.config['thresholds']['performance_decay']
                }
                
                # Экспорт в Prometheus
                self.model_performance_decay.labels(
                    model_version=self.config['model']['version'],
                    metric=metric_name
                ).set(decay)
        
        # Расчет общего дрифта концепта
        target_drift_detected = target_results['metrics'][0]['result']['target_drift']
        target_drift_score = target_results['metrics'][0]['result'].get('drift_score', 0)
        
        # Учет дрифта в прогнозах (если есть столбец предсказаний)
        if 'prediction' in current_data.columns and 'prediction' in self.reference_data.columns:
            prediction_drift_report = Report(metrics=[
                ColumnDriftMetric(column_name='prediction')
            ])
            
            prediction_drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            prediction_drift = prediction_drift_report.as_dict()
            prediction_drift_score = prediction_drift['metrics'][0]['result']['drift_score']
        else:
            prediction_drift_score = 0
        
        # Комбинированный score дрифта концепта
        concept_drift_score = max(target_drift_score, prediction_drift_score)
        
        self.concept_drift_score.labels(
            model_version=self.config['model']['version']
        ).set(concept_drift_score)
        
        # Сохранение отчетов
        report_path = Path(f"monitoring/reports/concept_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        target_drift_report.save_html(str(report_path))
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'concept_drift_score': concept_drift_score,
            'target_drift_detected': target_drift_detected,
            'target_drift_score': target_drift_score,
            'prediction_drift_score': prediction_drift_score,
            'performance_decay': performance_decay,
            'significant_performance_decay': any(
                decay_info['significant_decay'] 
                for decay_info in performance_decay.values()
            ),
            'report_path': str(report_path),
            'concept_drift_detected': concept_drift_score > self.config['thresholds']['concept_drift']
        }
        
        # Логирование в MLflow
        with mlflow.start_run(run_name=f"concept_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_metrics({
                'concept_drift_score': concept_drift_score,
                'target_drift_score': target_drift_score,
                'prediction_drift_score': prediction_drift_score
            })
            
            for metric_name, decay_info in performance_decay.items():
                mlflow.log_metric(f"decay_{metric_name}", decay_info['decay_percent'])
            
            mlflow.log_artifact(str(report_path))
            mlflow.log_dict(results, 'concept_drift_results.json')
        
        return results
    
    def _get_reference_performance(self) -> Dict[str, float]:
        """Получение референсной производительности модели"""
        metrics_path = Path(self.config['model']['reference_metrics_path'])
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        else:
            # Возвращаем дефолтные значения, если файл не найден
            return {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.79,
                'f1': 0.80,
                'roc_auc': 0.88
            }
    
    def run_tests(self, current_data: pd.DataFrame) -> TestSuite:
        """Запуск тестов для проверки качества данных"""
        self.logger.info("Running data quality tests...")
        
        test_suite = TestSuite(tests=[
            TestShareOfDriftedColumns(
                columns=self.column_mapping.numerical_features + self.column_mapping.categorical_features,
                lt=self.config['thresholds']['max_drifted_columns_share']
            ),
            TestNumberOfDriftedColumns(
                columns=self.column_mapping.numerical_features + self.column_mapping.categorical_features,
                lt=int(self.config['thresholds']['max_drifted_columns_count'])
            ),
            TestShareOfMissingValues(
                columns=self.column_mapping.numerical_features,
                lt=self.config['thresholds']['max_missing_values_share']
            ),
        ])
        
        # Добавление тестов для важных фич
        important_features = self.config['data'].get('important_features', [])
        for feature in important_features:
            if feature in self.column_mapping.numerical_features:
                test_suite.add_test(TestColumnDrift(column_name=feature))
                
                # Тесты на диапазон значений
                if f"{feature}_min" in self.config['thresholds']:
                    test_suite.add_test(TestColumnValueMin(
                        column_name=feature,
                        gt=self.config['thresholds'][f"{feature}_min"]
                    ))
                
                if f"{feature}_max" in self.config['thresholds']:
                    test_suite.add_test(TestColumnValueMax(
                        column_name=feature,
                        lt=self.config['thresholds'][f"{feature}_max"]
                    ))
        
        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Сохранение результатов тестов
        test_results_path = Path(f"monitoring/reports/tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        test_suite.save_html(str(test_results_path))
        
        return test_suite
    
    def check_alerts(self, data_drift_results: Dict, concept_drift_results: Dict) -> List[Dict]:
        """Проверка условий для алертов"""
        alerts = []
        
        # Алерты на дрифт данных
        if data_drift_results.get('drift_detected', False):
            alerts.append({
                'level': 'WARNING' if data_drift_results['dataset_drift_score'] < 0.3 else 'CRITICAL',
                'type': 'DATA_DRIFT',
                'message': f"Data drift detected. Score: {data_drift_results['dataset_drift_score']:.3f}",
                'details': {
                    'drifted_columns': data_drift_results['drifted_columns'],
                    'score': data_drift_results['dataset_drift_score']
                },
                'timestamp': datetime.now().isoformat()
            })
            
            self.drift_alerts_total.labels(
                alert_level='WARNING' if data_drift_results['dataset_drift_score'] < 0.3 else 'CRITICAL',
                model_version=self.config['model']['version']
            ).inc()
        
        # Алерты на дрифт концепта
        if concept_drift_results.get('concept_drift_detected', False):
            alerts.append({
                'level': 'WARNING' if concept_drift_results['concept_drift_score'] < 0.3 else 'CRITICAL',
                'type': 'CONCEPT_DRIFT',
                'message': f"Concept drift detected. Score: {concept_drift_results['concept_drift_score']:.3f}",
                'details': concept_drift_results.get('performance_decay', {}),
                'timestamp': datetime.now().isoformat()
            })
            
            self.drift_alerts_total.labels(
                alert_level='WARNING' if concept_drift_results['concept_drift_score'] < 0.3 else 'CRITICAL',
                model_version=self.config['model']['version']
            ).inc()
        
        # Алерты на деградацию производительности
        if concept_drift_results.get('significant_performance_decay', False):
            decay_info = concept_drift_results.get('performance_decay', {})
            for metric_name, metric_info in decay_info.items():
                if metric_info.get('significant_decay', False):
                    alerts.append({
                        'level': 'CRITICAL',
                        'type': 'PERFORMANCE_DECAY',
                        'message': f"Significant performance decay in {metric_name}: {metric_info['decay_percent']:.1f}%",
                        'details': metric_info,
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Логирование алертов
        if alerts:
            alerts_path = Path(f"monitoring/alerts/alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            alerts_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(alerts_path, 'w') as f:
                json.dump(alerts, f, indent=2)
            
            self.logger.warning(f"Generated {len(alerts)} alerts")
        
        return alerts
    
    def generate_dashboard_data(self, data_drift_results: Dict, concept_drift_results: Dict) -> Dict:
        """Генерация данных для дашборда"""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': {
                'score': data_drift_results.get('dataset_drift_score', 0),
                'drifted_columns_count': len(data_drift_results.get('drifted_columns', [])),
                'total_columns': len(self.column_mapping.numerical_features + self.column_mapping.categorical_features),
                'status': 'OK' if not data_drift_results.get('drift_detected', False) else 'WARNING'
            },
            'concept_drift': {
                'score': concept_drift_results.get('concept_drift_score', 0),
                'target_drift': concept_drift_results.get('target_drift_detected', False),
                'performance_decay': concept_drift_results.get('performance_decay', {}),
                'status': 'OK' if not concept_drift_results.get('concept_drift_detected', False) else 'WARNING'
            },
            'data_quality': {
                'current_samples': data_drift_results.get('total_samples_current', 0),
                'reference_samples': data_drift_results.get('total_samples_reference', 0),
                'coverage_percentage': (data_drift_results.get('total_samples_current', 0) / 
                                      max(data_drift_results.get('total_samples_reference', 1), 1)) * 100
            }
        }
        
        # Сохранение данных для Grafana
        dashboard_path = Path(f"monitoring/dashboard/dashboard_{datetime.now().strftime('%Y%m%d')}.json")
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        return dashboard_data
    
    def trigger_retraining(self, data_drift_results: Dict, concept_drift_results: Dict) -> bool:
        """Проверка условий для запуска переобучения"""
        should_retrain = False
        reasons = []
        
        # Условия для переобучения
        if data_drift_results.get('drift_detected', False):
            if data_drift_results['dataset_drift_score'] > self.config['thresholds']['retrain_data_drift']:
                should_retrain = True
                reasons.append(f"Data drift score {data_drift_results['dataset_drift_score']:.3f} exceeds threshold")
        
        if concept_drift_results.get('concept_drift_detected', False):
            if concept_drift_results['concept_drift_score'] > self.config['thresholds']['retrain_concept_drift']:
                should_retrain = True
                reasons.append(f"Concept drift score {concept_drift_results['concept_drift_score']:.3f} exceeds threshold")
        
        if concept_drift_results.get('significant_performance_decay', False):
            should_retrain = True
            reasons.append("Significant performance decay detected")
        
        if should_retrain:
            self.logger.info(f"Triggering retraining. Reasons: {', '.join(reasons)}")
            
            # Логирование решения
            retrain_decision = {
                'timestamp': datetime.now().isoformat(),
                'decision': 'RETRAIN',
                'reasons': reasons,
                'data_drift_score': data_drift_results.get('dataset_drift_score', 0),
                'concept_drift_score': concept_drift_results.get('concept_drift_score', 0),
                'thresholds': self.config['thresholds']
            }
            
            decision_path = Path(f"monitoring/retraining/decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            decision_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(decision_path, 'w') as f:
                json.dump(retrain_decision, f, indent=2)
            
            # Вызов API для запуска переобучения (опционально)
            if self.config.get('enable_auto_retraining', False):
                self._call_retraining_api()
        
        return should_retrain
    
    def _call_retraining_api(self):
        """Вызов API для запуска переобучения"""
        import requests
        
        try:
            response = requests.post(
                self.config['retraining']['api_endpoint'],
                json={
                    'trigger': 'drift_detected',
                    'timestamp': datetime.now().isoformat(),
                    'priority': 'high'
                },
                headers={'Authorization': f"Bearer {self.config['retraining']['api_token']}"},
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info("Retraining API called successfully")
            else:
                self.logger.error(f"Failed to call retraining API: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error calling retraining API: {str(e)}")
    
    def monitor(self, hours: int = 24) -> Dict:
        """Основной метод мониторинга"""
        self.logger.info(f"Starting drift monitoring for last {hours} hours")
        
        try:
            # 1. Сбор текущих данных
            current_data = self.collect_current_data(hours)
            
            if len(current_data) < self.config['min_samples_for_monitoring']:
                self.logger.warning(f"Insufficient samples: {len(current_data)}. Skipping monitoring.")
                return {'status': 'INSUFFICIENT_DATA', 'samples': len(current_data)}
            
            # 2. Расчет дрифта данных
            data_drift_results = self.calculate_data_drift(current_data)
            
            # 3. Расчет дрифта концепта
            concept_drift_results = self.calculate_concept_drift(current_data)
            
            # 4. Запуск тестов
            test_suite = self.run_tests(current_data)
            
            # 5. Проверка алертов
            alerts = self.check_alerts(data_drift_results, concept_drift_results)
            
            # 6. Генерация данных для дашборда
            dashboard_data = self.generate_dashboard_data(data_drift_results, concept_drift_results)
            
            # 7. Проверка необходимости переобучения
            should_retrain = self.trigger_retraining(data_drift_results, concept_drift_results)
            
            # 8. Сводный отчет
            summary = {
                'status': 'COMPLETED',
                'timestamp': datetime.now().isoformat(),
                'monitoring_period_hours': hours,
                'samples_analyzed': len(current_data),
                'data_drift_detected': data_drift_results.get('drift_detected', False),
                'concept_drift_detected': concept_drift_results.get('concept_drift_detected', False),
                'alerts_generated': len(alerts),
                'should_retrain': should_retrain,
                'data_drift_score': data_drift_results.get('dataset_drift_score', 0),
                'concept_drift_score': concept_drift_results.get('concept_drift_score', 0),
                'report_paths': {
                    'data_drift': data_drift_results.get('report_path'),
                    'concept_drift': concept_drift_results.get('report_path')
                }
            }
            
            # Сохранение сводного отчета
            summary_path = Path(f"monitoring/summaries/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Monitoring completed. Summary: {summary}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error during monitoring: {str(e)}", exc_info=True)
            
            error_report = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': str(e.__traceback__)
            }
            
            return error_report
    
    def start_prometheus_server(self, port: int = 8001):
        """Запуск сервера Prometheus для экспорта метрик"""
        start_http_server(port)
        self.logger.info(f"Prometheus metrics server started on port {port}")

if __name__ == "__main__":
    # Пример использования
    monitor = DriftMonitor()
    
    # Запуск сервера метрик
    monitor.start_prometheus_server()
    
    # Ежедневный мониторинг
    import schedule
    import time
    
    def daily_monitoring():
        results = monitor.monitor(hours=24)
        print(f"Daily monitoring completed: {results}")
    
    # Настройка расписания
    schedule.every().day.at("02:00").do(daily_monitoring)
    
    # Также запуск при старте
    daily_monitoring()
    
    # Бесконечный цикл для schedule
    while True:
        schedule.run_pending()
        time.sleep(60)