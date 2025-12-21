"""
Дашборд для мониторинга с использованием Evidently AI
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml

from evidently.report import Report
from evidently.metrics import *
from evidently.metric_preset import *
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import (
    DataDriftTab,
    DataQualityTab,
    RegressionPerformanceTab,
    ClassificationPerformanceTab,
    CatTargetDriftTab,
    NumTargetDriftTab
)

from src.utils.logger import monitoring_logger
from src.utils.database import get_database_manager
from src.utils.config_loader import get_config

class EvidentlyDashboard:
    """Класс для создания дашбордов мониторинга с Evidently AI"""
    
    def __init__(self, config_path: str = 'configs/monitoring_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Загрузка референсных данных
        self.reference_data = self._load_reference_data()
        
        # Конфигурация колонок
        self.column_mapping = ColumnMapping(
            target=self.config['data']['target_column'],
            prediction='prediction',
            numerical_features=self.config['data']['numerical_features'],
            categorical_features=self.config['data']['categorical_features'],
            datetime_feature=self.config['data'].get('datetime_feature'),
            task='classification'
        )
        
        monitoring_logger.info("EvidentlyDashboard initialized")
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Загрузка референсных данных"""
        ref_path = Path(self.config['data']['reference_path'])
        
        if ref_path.exists():
            data = pd.read_csv(ref_path)
            monitoring_logger.info(f"Loaded reference data: {data.shape}")
            return data
        else:
            monitoring_logger.warning(f"Reference data not found: {ref_path}")
            return pd.DataFrame()
    
    def collect_current_data(self, hours: int = 24) -> pd.DataFrame:
        """Сбор текущих данных из БД"""
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        query = f"""
        SELECT 
            features,
            prediction_result,
            created_at
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '{hours} hours'
        ORDER BY created_at DESC
        LIMIT 10000
        """
        
        try:
            results = db_pool.execute_query(query)
            
            if not results:
                monitoring_logger.warning(f"No current data found for last {hours} hours")
                return pd.DataFrame()
            
            # Преобразование в DataFrame
            data_rows = []
            for row in results:
                features = row['features']
                prediction = row['prediction_result'].get('probability', 0.5)
                data_row = {**features, 'prediction': prediction}
                data_rows.append(data_row)
            
            current_data = pd.DataFrame(data_rows)
            monitoring_logger.info(f"Collected current data: {current_data.shape}")
            
            return current_data
            
        except Exception as e:
            monitoring_logger.error(f"Failed to collect current data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def create_data_drift_dashboard(self, current_data: pd.DataFrame) -> Dashboard:
        """Создание дашборда для мониторинга дрифта данных"""
        monitoring_logger.info("Creating data drift dashboard")
        
        dashboard = Dashboard(tabs=[DataDriftTab(), DataQualityTab()])
        
        dashboard.calculate(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        return dashboard
    
    def create_target_drift_dashboard(self, current_data: pd.DataFrame) -> Dashboard:
        """Создание дашборда для мониторинга дрифта целевой переменной"""
        if self.column_mapping.target not in current_data.columns:
            monitoring_logger.warning("Target column not in current data")
            return None
        
        monitoring_logger.info("Creating target drift dashboard")
        
        # Определяем тип таргета
        if self.config['data'].get('target_type') == 'numerical':
            dashboard = Dashboard(tabs=[NumTargetDriftTab()])
        else:
            dashboard = Dashboard(tabs=[CatTargetDriftTab()])
        
        dashboard.calculate(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        return dashboard
    
    def create_performance_dashboard(self, current_data: pd.DataFrame) -> Dashboard:
        """Создание дашборда для мониторинга производительности модели"""
        if self.column_mapping.target not in current_data.columns:
            monitoring_logger.warning("Target column not in current data")
            return None
        
        monitoring_logger.info("Creating performance dashboard")
        
        if self.config['data'].get('target_type') == 'numerical':
            dashboard = Dashboard(tabs=[RegressionPerformanceTab()])
        else:
            dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
        
        dashboard.calculate(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        return dashboard
    
    def create_comprehensive_dashboard(self, current_data: pd.DataFrame) -> Dashboard:
        """Создание комплексного дашборда"""
        monitoring_logger.info("Creating comprehensive dashboard")
        
        dashboard = Dashboard(tabs=[
            DataDriftTab(),
            DataQualityTab(),
            ClassificationPerformanceTab(),
            CatTargetDriftTab()
        ])
        
        dashboard.calculate(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        return dashboard
    
    def save_dashboard(self, dashboard: Dashboard, filename: str):
        """Сохранение дашборда в HTML файл"""
        output_dir = Path("monitoring/dashboards")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        dashboard.save(str(filepath))
        
        monitoring_logger.info(f"Dashboard saved: {filepath}")
        
        return filepath
    
    def generate_daily_report(self):
        """Генерация ежедневного отчета"""
        monitoring_logger.info("Generating daily monitoring report")
        
        # Сбор данных за последние 24 часа
        current_data = self.collect_current_data(hours=24)
        
        if current_data.empty:
            monitoring_logger.warning("No data for daily report")
            return None
        
        # Создание дашбордов
        dashboards = {}
        
        # 1. Data Drift Dashboard
        data_drift_dashboard = self.create_data_drift_dashboard(current_data)
        if data_drift_dashboard:
            data_drift_file = self.save_dashboard(
                data_drift_dashboard,
                f"data_drift_{datetime.now().strftime('%Y%m%d')}.html"
            )
            dashboards['data_drift'] = str(data_drift_file)
        
        # 2. Target Drift Dashboard (если есть таргет)
        if self.column_mapping.target in current_data.columns:
            target_drift_dashboard = self.create_target_drift_dashboard(current_data)
            if target_drift_dashboard:
                target_drift_file = self.save_dashboard(
                    target_drift_dashboard,
                    f"target_drift_{datetime.now().strftime('%Y%m%d')}.html"
                )
                dashboards['target_drift'] = str(target_drift_file)
            
            # 3. Performance Dashboard
            performance_dashboard = self.create_performance_dashboard(current_data)
            if performance_dashboard:
                performance_file = self.save_dashboard(
                    performance_dashboard,
                    f"performance_{datetime.now().strftime('%Y%m%d')}.html"
                )
                dashboards['performance'] = str(performance_file)
        
        # 4. Comprehensive Dashboard
        comprehensive_dashboard = self.create_comprehensive_dashboard(current_data)
        if comprehensive_dashboard:
            comprehensive_file = self.save_dashboard(
                comprehensive_dashboard,
                f"comprehensive_{datetime.now().strftime('%Y%m%d')}.html"
            )
            dashboards['comprehensive'] = str(comprehensive_file)
        
        # Генерация отчета
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_period_hours': 24,
            'reference_samples': len(self.reference_data),
            'current_samples': len(current_data),
            'dashboards': dashboards,
            'summary': self._generate_summary(current_data)
        }
        
        # Сохранение отчета
        report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        monitoring_logger.info(f"Daily report saved: {report_path}")
        
        return report
    
    def _generate_summary(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Генерация сводки по мониторингу"""
        summary = {}
        
        if current_data.empty:
            return summary
        
        # Анализ дрифта данных
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        data_drift_result = data_drift_report.as_dict()
        summary['data_drift'] = {
            'drift_score': data_drift_result['metrics'][0]['result']['dataset_drift'],
            'drifted_columns': data_drift_result['metrics'][0]['result']['number_of_drifted_columns'],
            'share_of_drifted_columns': data_drift_result['metrics'][0]['result']['share_of_drifted_columns']
        }
        
        # Анализ качества данных
        data_quality_report = Report(metrics=[DataQualityPreset()])
        data_quality_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        data_quality_result = data_quality_report.as_dict()
        summary['data_quality'] = {
            'missing_values': data_quality_result['metrics'][0]['result']['current']['share_of_missing_values'],
            'new_values': data_quality_result['metrics'][1]['result']['current']['number_of_new_values']
        }
        
        # Анализ производительности (если есть таргет)
        if self.column_mapping.target in current_data.columns:
            performance_report = Report(metrics=[ClassificationPreset()])
            performance_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            performance_result = performance_report.as_dict()
            summary['performance'] = performance_result['metrics'][0]['result']['metrics']
        
        return summary
    
    def create_custom_metrics_report(self, current_data: pd.DataFrame, 
                                   metrics_config: Dict[str, Any]) -> Report:
        """Создание отчета с пользовательскими метриками"""
        monitoring_logger.info("Creating custom metrics report")
        
        metrics = []
        
        # Добавление метрик из конфигурации
        for metric_config in metrics_config.get('metrics', []):
            metric_type = metric_config.get('type')
            metric_params = metric_config.get('params', {})
            
            if metric_type == 'column_drift':
                metric = ColumnDriftMetric(column_name=metric_params.get('column_name'))
            elif metric_type == 'column_summary':
                metric = ColumnSummaryMetric(column_name=metric_params.get('column_name'))
            elif metric_type == 'column_correlations':
                metric = ColumnCorrelationsMetric(column_name=metric_params.get('column_name'))
            elif metric_type == 'column_value_range':
                metric = ColumnValueRangeMetric(
                    column_name=metric_params.get('column_name'),
                    left=metric_params.get('left'),
                    right=metric_params.get('right')
                )
            elif metric_type == 'column_quantile':
                metric = ColumnQuantileMetric(
                    column_name=metric_params.get('column_name'),
                    quantile=metric_params.get('quantile', 0.5)
                )
            else:
                monitoring_logger.warning(f"Unknown metric type: {metric_type}")
                continue
            
            metrics.append(metric)
        
        # Создание отчета
        report = Report(metrics=metrics)
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        return report
    
    def stream_dashboard_updates(self, update_interval: int = 3600):
        """Потоковое обновление дашбордов"""
        import time
        import threading
        
        def update_loop():
            while True:
                try:
                    self.generate_daily_report()
                    monitoring_logger.info(f"Dashboard updated at {datetime.now()}")
                except Exception as e:
                    monitoring_logger.error(f"Dashboard update failed: {e}", exc_info=True)
                
                time.sleep(update_interval)
        
        # Запуск в отдельном потоке
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
        monitoring_logger.info(f"Dashboard streaming started with {update_interval}s interval")
        
        return update_thread

# Утилиты для интеграции с веб-интерфейсом
def create_web_dashboard_config():
    """Создание конфигурации для веб-дашборда"""
    config = {
        'layout': {
            'title': 'Credit Scoring Monitoring Dashboard',
            'tabs': [
                {
                    'name': 'Data Drift',
                    'id': 'data_drift',
                    'description': 'Monitoring of feature distributions'
                },
                {
                    'name': 'Model Performance',
                    'id': 'performance',
                    'description': 'Model accuracy and metrics over time'
                },
                {
                    'name': 'Data Quality',
                    'id': 'data_quality',
                    'description': 'Data completeness and consistency'
                },
                {
                    'name': 'Predictions',
                    'id': 'predictions',
                    'description': 'Prediction statistics and trends'
                }
            ]
        },
        'metrics': {
            'data_drift': ['dataset_drift', 'drifted_columns_count'],
            'performance': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            'data_quality': ['missing_values', 'new_categories', 'range_violations']
        },
        'refresh_interval': 300,  # 5 минут
        'data_retention_days': 30
    }
    
    return config

if __name__ == "__main__":
    # Пример использования
    dashboard = EvidentlyDashboard()
    
    # Генерация ежедневного отчета
    report = dashboard.generate_daily_report()
    
    if report:
        print(f"Daily report generated: {report['timestamp']}")
        print(f"Dashboards created: {list(report['dashboards'].keys())}")
        
        # Запуск потокового обновления
        dashboard.stream_dashboard_updates(update_interval=3600)  # Каждый час