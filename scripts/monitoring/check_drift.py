"""
Мониторинг дрифта данных и концепта
Этап 6: Детектирование дрифта с использованием Evidently AI
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Evidently для мониторинга дрифта
try:
    from evidently.report import Report
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
        DataQualityTable,
    )
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.test_suite import TestSuite
    from evidently.tests import (
        TestNumberOfDriftedColumns,
        TestShareOfDriftedColumns,
        TestColumnDrift,
    )
except ImportError:
    print("Установите evidently: pip install evidently")
    exit(1)

class DriftMonitor:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.reference_data = None
        self.current_data = None
        self.results = {}
        
    def load_config(self, config_path):
        """Загрузка конфигурации"""
        default_config = {
            'thresholds': {
                'data_drift': 0.3,
                'concept_drift': 0.2,
                'max_drifted_columns_share': 0.3,
                'retrain_threshold': 0.5
            },
            'monitoring': {
                'window_size': 1000,
                'check_interval_hours': 24,
                'min_samples': 100
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Объединяем конфигурации
                if 'thresholds' in user_config:
                    default_config['thresholds'].update(user_config['thresholds'])
                if 'monitoring' in user_config:
                    default_config['monitoring'].update(user_config['monitoring'])
        
        return default_config
    
    def load_reference_data(self, reference_path, target_column='default'):
        """Загрузка эталонных данных"""
        print(f"Загрузка эталонных данных из {reference_path}")
        
        try:
            self.reference_data = pd.read_csv(reference_path)
            
            # Убедимся, что есть целевая переменная
            if target_column not in self.reference_data.columns:
                print(f"Предупреждение: целевая переменная '{target_column}' не найдена")
            
            print(f"Эталонные данные: {self.reference_data.shape[0]} строк, "
                  f"{self.reference_data.shape[1]} колонок")
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки эталонных данных: {e}")
            return False
    
    def load_current_data(self, current_path, target_column='default'):
        """Загрузка текущих данных"""
        print(f"Загрузка текущих данных из {current_path}")
        
        try:
            self.current_data = pd.read_csv(current_path)
            
            # Проверяем совместимость колонок
            ref_cols = set(self.reference_data.columns) if self.reference_data is not None else set()
            cur_cols = set(self.current_data.columns)
            
            if ref_cols and not cur_cols.issubset(ref_cols):
                print(f"Предупреждение: текущие данные содержат новые колонки: "
                      f"{cur_cols - ref_cols}")
            
            print(f"Текущие данные: {self.current_data.shape[0]} строк, "
                  f"{self.current_data.shape[1]} колонок")
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки текущих данных: {e}")
            return False
    
    def check_data_drift(self):
        """Проверка дрифта данных"""
        print("\nПроверка дрифта данных...")
        
        if self.reference_data is None or self.current_data is None:
            print("Ошибка: данные не загружены")
            return None
        
        # Проверяем минимальное количество образцов
        min_samples = self.config['monitoring']['min_samples']
        if len(self.current_data) < min_samples:
            print(f"Предупреждение: недостаточно данных ({len(self.current_data)} < {min_samples})")
            # Можно использовать накопленные данные, но для простоты пропустим
            return None
        
        # Создаем отчет о дрифте данных
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityTable(),
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )
        
        # Получаем результаты
        report_result = data_drift_report.as_dict()
        
        # Извлекаем метрики дрифта
        drift_metrics = {}
        
        for metric in report_result['metrics']:
            if metric['metric'] == 'DatasetDriftMetric':
                drift_metrics['dataset_drift'] = metric['result']['dataset_drift']
                drift_metrics['drifted_columns'] = metric['result']['drifted_columns']
                drift_metrics['number_of_drifted_columns'] = metric['result']['number_of_drifted_columns']
                drift_metrics['share_of_drifted_columns'] = metric['result']['share_of_drifted_columns']
            elif metric['metric'] == 'DataDriftTable':
                drift_metrics['column_drifts'] = {}
                for col_name, col_result in metric['result'].items():
                    if isinstance(col_result, dict) and 'drift_score' in col_result:
                        drift_metrics['column_drifts'][col_name] = {
                            'drift_score': col_result['drift_score'],
                            'drift_detected': col_result['drift_detected']
                        }
        
        # Определяем, нужно ли переобучать модель
        threshold = self.config['thresholds']['data_drift']
        retrain_threshold = self.config['thresholds']['retrain_threshold']
        
        should_alert = False
        should_retrain = False
        
        if 'share_of_drifted_columns' in drift_metrics:
            share_drifted = drift_metrics['share_of_drifted_columns']
            
            if share_drifted > threshold:
                should_alert = True
                print(f"Обнаружен дрифт данных: {share_drifted:.2%} колонок с дрифтом "
                      f"(порог: {threshold:.2%})")
            
            if share_drifted > retrain_threshold:
                should_retrain = True
                print(f"Требуется переобучение модели: {share_drifted:.2%} колонок с дрифтом "
                      f"(порог переобучения: {retrain_threshold:.2%})")
        
        # Сохраняем результаты
        self.results['data_drift'] = {
            'metrics': drift_metrics,
            'should_alert': should_alert,
            'should_retrain': should_retrain,
            'check_timestamp': datetime.now().isoformat()
        }
        
        return self.results['data_drift']
    
    def check_concept_drift(self, model=None, reference_predictions=None, current_predictions=None):
        """Проверка дрифта концепта (целевой переменной)"""
        print("\nПроверка дрифта концепта...")
        
        if self.reference_data is None or self.current_data is None:
            print("Ошибка: данные не загружены")
            return None
        
        # Проверяем наличие целевой переменной
        target_col = 'default'
        if target_col not in self.reference_data.columns or target_col not in self.current_data.columns:
            print(f"Предупреждение: целевая переменная '{target_col}' не найдена в данных")
            return None
        
        # Создаем отчет о дрифте целевой переменной
        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        target_drift_report.run(
            reference_data=self.reference_data[[target_col]],
            current_data=self.current_data[[target_col]]
        )
        
        # Получаем результаты
        report_result = target_drift_report.as_dict()
        
        # Извлекаем метрики дрифта концепта
        concept_drift_metrics = {}
        
        for metric in report_result['metrics']:
            if metric['metric'] == 'ColumnDriftMetric' and metric['result']['column_name'] == target_col:
                concept_drift_metrics['drift_score'] = metric['result']['drift_score']
                concept_drift_metrics['drift_detected'] = metric['result']['drift_detected']
                concept_drift_metrics['current': {
                    'mean': metric['result']['current']['distribution']['mean'],
                    'std': metric['result']['current']['distribution']['std']
                }]
                concept_drift_metrics['reference': {
                    'mean': metric['result']['reference']['distribution']['mean'],
                    'std': metric['result']['reference']['distribution']['std']
                }]
                break
        
        # Определяем, нужно ли переобучать модель
        threshold = self.config['thresholds']['concept_drift']
        
        should_alert = False
        should_retrain = False
        
        if 'drift_score' in concept_drift_metrics:
            drift_score = concept_drift_metrics['drift_score']
            
            if drift_score > threshold:
                should_alert = True
                print(f"Обнаружен дрифт концепта: score={drift_score:.3f} "
                      f"(порог: {threshold:.3f})")
                should_retrain = True
        
        # Сохраняем результаты
        self.results['concept_drift'] = {
            'metrics': concept_drift_metrics,
            'should_alert': should_alert,
            'should_retrain': should_retrain,
            'check_timestamp': datetime.now().isoformat()
        }
        
        return self.results['concept_drift']
    
    def run_tests(self):
        """Запуск тестов для автоматических проверок"""
        print("\nЗапуск тестов...")
        
        if self.reference_data is None or self.current_data is None:
            print("Ошибка: данные не загружены")
            return None
        
        # Создаем набор тестов
        tests = TestSuite(tests=[
            TestNumberOfDriftedColumns(
                columns=self.reference_data.columns.tolist(),
                lt=int(len(self.reference_data.columns) * 
                      self.config['thresholds']['max_drifted_columns_share'])
            ),
            TestShareOfDriftedColumns(
                columns=self.reference_data.columns.tolist(),
                lt=self.config['thresholds']['data_drift']
            ),
        ])
        
        tests.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )
        
        # Получаем результаты тестов
        test_results = tests.as_dict()
        self.results['tests'] = test_results
        
        # Анализируем результаты тестов
        failed_tests = []
        for test in test_results['tests']:
            if not test['status'] == 'SUCCESS':
                failed_tests.append({
                    'name': test['name'],
                    'status': test['status'],
                    'description': test['description']
                })
        
        self.results['failed_tests'] = failed_tests
        self.results['all_tests_passed'] = len(failed_tests) == 0
        
        print(f"Тестов выполнено: {len(test_results['tests'])}")
        print(f"Проваленных тестов: {len(failed_tests)}")
        
        return self.results['tests']
    
    def generate_report(self, output_dir):
        """Генерация отчета"""
        print("\nГенерация отчета...")
        
        # Создаем директорию если не существует
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем JSON отчет
        json_report = {
            'drift_monitoring': self.results,
            'config': self.config,
            'summary': self.get_summary(),
            'timestamp': timestamp
        }
        
        json_path = output_path / f"drift_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Генерируем HTML отчет
        try:
            # Создаем визуальный отчет
            visual_report = Report(metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
            ])
            
            visual_report.run(
                reference_data=self.reference_data,
                current_data=self.current_data
            )
            
            html_path = output_path / f"drift_report_{timestamp}.html"
            visual_report.save_html(str(html_path))
            
            print(f"HTML отчет сохранен: {html_path}")
        except Exception as e:
            print(f"Не удалось создать HTML отчет: {e}")
        
        print(f"JSON отчет сохранен: {json_path}")
        
        return json_path, html_path if 'html_path' in locals() else None
    
    def get_summary(self):
        """Сводка результатов"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_drift_detected': False,
            'concept_drift_detected': False,
            'should_alert': False,
            'should_retrain': False,
            'reasons': []
        }
        
        # Проверяем дрифт данных
        if 'data_drift' in self.results:
            data_drift = self.results['data_drift']
            if data_drift.get('should_alert', False):
                summary['data_drift_detected'] = True
                summary['should_alert'] = True
                summary['reasons'].append('Дрифт данных')
            
            if data_drift.get('should_retrain', False):
                summary['should_retrain'] = True
                summary['reasons'].append('Высокий дрифт данных')
        
        # Проверяем дрифт концепта
        if 'concept_drift' in self.results:
            concept_drift = self.results['concept_drift']
            if concept_drift.get('should_alert', False):
                summary['concept_drift_detected'] = True
                summary['should_alert'] = True
                summary['reasons'].append('Дрифт концепта')
            
            if concept_drift.get('should_retrain', False):
                summary['should_retrain'] = True
                summary['reasons'].append('Дрифт концепта')
        
        # Проверяем тесты
        if 'failed_tests' in self.results:
            if len(self.results['failed_tests']) > 0:
                summary['should_alert'] = True
                summary['reasons'].append(f"Провалено тестов: {len(self.results['failed_tests'])}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Мониторинг дрифта данных и концепта')
    parser.add_argument('--reference-data', required=True,
                       help='Путь к эталонным данным (train.csv)')
    parser.add_argument('--current-data', required=True,
                       help='Путь к текущим данным (latest_production_data.csv)')
    parser.add_argument('--config', default='configs/monitoring_config.yaml',
                       help='Файл конфигурации')
    parser.add_argument('--output-dir', default='reports/drift_monitoring',
                       help='Директория для сохранения отчетов')
    parser.add_argument('--run-tests', action='store_true',
                       help='Запуск автоматических тестов')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("МОНИТОРИНГ ДРИФТА ДАННЫХ И КОНЦЕПТА")
    print("=" * 60)
    
    # Инициализация монитора
    monitor = DriftMonitor(args.config)
    
    # Загрузка данных
    if not monitor.load_reference_data(args.reference_data):
        print("Не удалось загрузить эталонные данные")
        return
    
    if not monitor.load_current_data(args.current_data):
        print("Не удалось загрузить текущие данные")
        return
    
    # Проверка дрифта данных
    data_drift_result = monitor.check_data_drift()
    
    # Проверка дрифта концепта
    concept_drift_result = monitor.check_concept_drift()
    
    # Запуск тестов если нужно
    if args.run_tests:
        monitor.run_tests()
    
    # Генерация отчетов
    json_report, html_report = monitor.generate_report(args.output_dir)
    
    # Вывод сводки
    summary = monitor.get_summary()
    
    print("\n" + "=" * 60)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    print(f"Дрифт данных обнаружен: {'Да' if summary['data_drift_detected'] else 'Нет'}")
    print(f"Дрифт концепта обнаружен: {'Да' if summary['concept_drift_detected'] else 'Нет'}")
    print(f"Требуется алерт: {'Да' if summary['should_alert'] else 'Нет'}")
    print(f"Требуется переобучение: {'Да' if summary['should_retrain'] else 'Нет'}")
    
    if summary['reasons']:
        print(f"Причины: {', '.join(summary['reasons'])}")
    
    print(f"\nОтчеты сохранены в: {args.output_dir}")
    
    # Для CI/CD: возвращаем код выхода в зависимости от результатов
    if summary['should_retrain']:
        print("\nРекомендация: запустить переобучение модели")
        # В CI/CD можно установить специальный флаг или выйти с кодом 1
        # exit(1)

if __name__ == "__main__":
    main()