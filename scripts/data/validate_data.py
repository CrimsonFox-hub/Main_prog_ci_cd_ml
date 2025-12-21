"""
Валидация данных для кредитного скоринга
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Загрузка конфигурации валидации"""
        default_config = {
            'validation': {
                'min_samples': 100,
                'max_missing_percent': 5,
                'numeric_ranges': {
                    'age': (18, 100),
                    'income': (0, 1000000),
                    'credit_amount': (0, 1000000),
                    'loan_duration': (1, 120),
                    'payment_to_income': (0, 1),
                    'existing_credits': (0, 10),
                    'dependents': (0, 10),
                    'residence_since': (0, 50),
                    'installment_rate': (1, 4)
                },
                'categorical_values': {
                    'credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
                    'purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
                    'savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
                    'employment': ['A71', 'A72', 'A73', 'A74', 'A75'],
                    'property': ['A121', 'A122', 'A123', 'A124']
                },
                'target_column': 'default',
                'target_values': [0, 1]
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if 'validation' in user_config:
                    default_config['validation'].update(user_config['validation'])
        
        return default_config
    
    def validate_dataset(self, data_path: str, dataset_type: str = 'train') -> Dict:
        """Основная функция валидации датасета"""
        print(f"Валидация датасета: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            self.validation_results[dataset_type] = {
                'file_path': data_path,
                'rows': len(df),
                'columns': len(df.columns),
                'timestamp': datetime.now().isoformat()
            }
            
            # Выполнение всех проверок
            self.validate_structure(df, dataset_type)
            self.validate_missing_values(df, dataset_type)
            self.validate_numeric_ranges(df, dataset_type)
            self.validate_categorical_values(df, dataset_type)
            self.validate_target_variable(df, dataset_type)
            self.validate_data_quality(df, dataset_type)
            
            # Сводка
            self.generate_summary(dataset_type)
            
            return self.validation_results[dataset_type]
            
        except Exception as e:
            error_msg = f"Ошибка при валидации {dataset_type}: {str(e)}"
            self.errors.append(error_msg)
            print(error_msg)
            return {'error': error_msg}
    
    def validate_structure(self, df: pd.DataFrame, dataset_type: str):
        """Проверка структуры данных"""
        print(f"  Проверка структуры данных...")
        
        checks = {
            'min_samples': len(df) >= self.config['validation']['min_samples'],
            'has_target_column': self.config['validation']['target_column'] in df.columns,
            'has_features': len(df.columns) > 1
        }
        
        self.validation_results[dataset_type]['structure_checks'] = checks
        
        if not checks['min_samples']:
            self.warnings.append(f"{dataset_type}: Мало образцов ({len(df)} < {self.config['validation']['min_samples']})")
        
        if not checks['has_target_column']:
            self.errors.append(f"{dataset_type}: Отсутствует целевая переменная")
    
    def validate_missing_values(self, df: pd.DataFrame, dataset_type: str):
        """Проверка пропущенных значений"""
        print(f"  Проверка пропущенных значений...")
        
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df)) * 100
        
        problematic_columns = missing_percent[missing_percent > 0].to_dict()
        
        self.validation_results[dataset_type]['missing_values'] = {
            'total_missing': missing_stats.sum(),
            'missing_columns': problematic_columns,
            'max_missing_percent': missing_percent.max() if len(missing_percent) > 0 else 0
        }
        
        # Проверка на превышение лимита пропущенных значений
        max_allowed = self.config['validation']['max_missing_percent']
        for col, percent in problematic_columns.items():
            if percent > max_allowed:
                self.errors.append(
                    f"{dataset_type}: Слишком много пропущенных значений в '{col}': {percent:.1f}% > {max_allowed}%"
                )
            elif percent > 0:
                self.warnings.append(
                    f"{dataset_type}: Пропущенные значения в '{col}': {percent:.1f}%"
                )
    
    def validate_numeric_ranges(self, df: pd.DataFrame, dataset_type: str):
        """Проверка числовых диапазонов"""
        print(f"  Проверка числовых диапазонов...")
        
        numeric_ranges = self.config['validation']['numeric_ranges']
        violations = {}
        
        for column, (min_val, max_val) in numeric_ranges.items():
            if column in df.columns:
                # Проверка на числовой тип
                if pd.api.types.is_numeric_dtype(df[column]):
                    out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
                    
                    if not out_of_range.empty:
                        violations[column] = {
                            'min_allowed': min_val,
                            'max_allowed': max_val,
                            'actual_min': df[column].min(),
                            'actual_max': df[column].max(),
                            'violations_count': len(out_of_range),
                            'violation_percent': (len(out_of_range) / len(df)) * 100,
                            'examples': out_of_range[column].head(5).tolist()
                        }
                        
                        if violations[column]['violation_percent'] > 5:
                            self.errors.append(
                                f"{dataset_type}: Много значений вне диапазона в '{column}': "
                                f"{violations[column]['violation_percent']:.1f}%"
                            )
                        else:
                            self.warnings.append(
                                f"{dataset_type}: Значения вне диапазона в '{column}'"
                            )
        
        self.validation_results[dataset_type]['numeric_range_violations'] = violations
    
    def validate_categorical_values(self, df: pd.DataFrame, dataset_type: str):
        """Проверка категориальных значений"""
        print(f"  Проверка категориальных значений...")
        
        categorical_rules = self.config['validation']['categorical_values']
        violations = {}
        
        for column, allowed_values in categorical_rules.items():
            if column in df.columns:
                # Проверка на строковый тип
                if pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
                    unique_values = df[column].unique()
                    invalid_values = [val for val in unique_values if val not in allowed_values]
                    
                    if invalid_values:
                        violations[column] = {
                            'allowed_values': allowed_values,
                            'invalid_values': invalid_values,
                            'invalid_count': df[column].isin(invalid_values).sum(),
                            'invalid_percent': (df[column].isin(invalid_values).sum() / len(df)) * 100
                        }
                        
                        if violations[column]['invalid_percent'] > 1:
                            self.errors.append(
                                f"{dataset_type}: Неверные категориальные значения в '{column}': "
                                f"{violations[column]['invalid_percent']:.1f}%"
                            )
                        else:
                            self.warnings.append(
                                f"{dataset_type}: Найдены нестандартные значения в '{column}'"
                            )
        
        self.validation_results[dataset_type]['categorical_violations'] = violations
    
    def validate_target_variable(self, df: pd.DataFrame, dataset_type: str):
        """Проверка целевой переменной"""
        print(f"  Проверка целевой переменной...")
        
        target_col = self.config['validation']['target_column']
        target_checks = {}
        
        if target_col in df.columns:
            target_series = df[target_col]
            
            # Проверка допустимых значений
            allowed_values = self.config['validation']['target_values']
            invalid_values = target_series[~target_series.isin(allowed_values)]
            
            if not invalid_values.empty:
                self.errors.append(
                    f"{dataset_type}: Неверные значения в целевой переменной: {invalid_values.unique()}"
                )
            
            # Проверка распределения
            value_counts = target_series.value_counts()
            value_percent = target_series.value_counts(normalize=True) * 100
            
            target_checks = {
                'has_target': True,
                'value_counts': value_counts.to_dict(),
                'value_percentages': value_percent.to_dict(),
                'class_imbalance': abs(value_percent[0] - value_percent[1]) if len(value_percent) == 2 else None,
                'missing_target': target_series.isnull().sum()
            }
            
            # Проверка дисбаланса классов
            if target_checks['class_imbalance'] and target_checks['class_imbalance'] > 40:
                self.warnings.append(
                    f"{dataset_type}: Сильный дисбаланс классов: {target_checks['class_imbalance']:.1f}%"
                )
            
            # Проверка пропущенных значений в целевой переменной
            if target_checks['missing_target'] > 0:
                self.errors.append(
                    f"{dataset_type}: Пропущенные значения в целевой переменной: {target_checks['missing_target']}"
                )
        else:
            target_checks = {'has_target': False}
            self.warnings.append(f"{dataset_type}: Целевая переменная не найдена")
        
        self.validation_results[dataset_type]['target_checks'] = target_checks
    
    def validate_data_quality(self, df: pd.DataFrame, dataset_type: str):
        """Проверка качества данных"""
        print(f"  Проверка качества данных...")
        
        quality_metrics = {}
        
        # Проверка на дубликаты
        duplicates = df.duplicated().sum()
        quality_metrics['duplicates'] = {
            'count': duplicates,
            'percent': (duplicates / len(df)) * 100
        }
        
        if duplicates > 0:
            self.warnings.append(
                f"{dataset_type}: Найдены дубликаты: {duplicates} ({quality_metrics['duplicates']['percent']:.1f}%)"
            )
        
        # Проверка на постоянные столбцы
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)
        
        quality_metrics['constant_columns'] = constant_columns
        
        if constant_columns:
            self.warnings.append(
                f"{dataset_type}: Постоянные столбцы: {constant_columns}"
            )
        
        # Проверка на выбросы (для числовых колонок)
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_percent = (outlier_count / len(df)) * 100
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': outlier_count,
                        'percent': outlier_percent,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    
                    if outlier_percent > 5:
                        self.warnings.append(
                            f"{dataset_type}: Много выбросов в '{col}': {outlier_percent:.1f}%"
                        )
        
        quality_metrics['outliers'] = outliers
        
        self.validation_results[dataset_type]['quality_metrics'] = quality_metrics
    
    def generate_summary(self, dataset_type: str):
        """Генерация сводки по валидации"""
        results = self.validation_results[dataset_type]
        
        # Подсчет количества проверок
        passed_checks = 0
        total_checks = 0
        
        # Структурные проверки
        if 'structure_checks' in results:
            total_checks += len(results['structure_checks'])
            passed_checks += sum(results['structure_checks'].values())
        
        # Проверка пропущенных значений
        if 'missing_values' in results:
            total_checks += 1
            if results['missing_values']['max_missing_percent'] <= self.config['validation']['max_missing_percent']:
                passed_checks += 1
        
        # Проверка целевой переменной
        if 'target_checks' in results and results['target_checks'].get('has_target', False):
            total_checks += 2  # наличие и распределение
            passed_checks += 1  # наличие
        
            if results['target_checks'].get('missing_target', 0) == 0:
                passed_checks += 1
        
        results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'validation_score': (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            'error_count': len([e for e in self.errors if dataset_type in e]),
            'warning_count': len([w for w in self.warnings if dataset_type in w]),
            'is_valid': len([e for e in self.errors if dataset_type in e]) == 0
        }
    
    def save_results(self, output_dir: str):
        """Сохранение результатов валидации"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение JSON отчета
        json_report = {
            'validation_results': self.validation_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'config': self.config,
            'timestamp': timestamp,
            'summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'datasets_validated': list(self.validation_results.keys()),
                'overall_status': 'PASS' if len(self.errors) == 0 else 'FAIL'
            }
        }
        
        json_path = output_path / f"data_validation_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Генерация HTML отчета
        html_path = output_path / f"data_validation_{timestamp}.html"
        self.generate_html_report(html_path, json_report)
        
        # Генерация текстового отчета
        txt_path = output_path / f"data_validation_{timestamp}.txt"
        self.generate_text_report(txt_path, json_report)
        
        print(f"\nОтчеты сохранены в: {output_dir}")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
        print(f"  Текст: {txt_path}")
        
        return json_path, html_path
    
    def generate_html_report(self, output_path: Path, report_data: Dict):
        """Генерация HTML отчета"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Отчет валидации данных - {report_data['timestamp']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; padding: 15px; background: #e8f4f8; border-radius: 5px; }}
                .dataset {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .error {{ color: #d32f2f; background: #ffebee; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                .warning {{ color: #f57c00; background: #fff3e0; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                .success {{ color: #388e3c; background: #e8f5e9; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Отчет валидации данных</h1>
                <p>Дата: {report_data['timestamp']}</p>
                <p>Статус: <span class="{'error' if report_data['summary']['overall_status'] == 'FAIL' else 'success'}">
                    {report_data['summary']['overall_status']}
                </span></p>
            </div>
            
            <div class="summary">
                <h2>Сводка</h2>
                <p>Ошибки: {report_data['summary']['total_errors']}</p>
                <p>Предупреждения: {report_data['summary']['total_warnings']}</p>
                <p>Датасеты: {', '.join(report_data['summary']['datasets_validated'])}</p>
            </div>
        """
        
        # Добавляем информацию по каждому датасету
        for dataset_name, results in report_data['validation_results'].items():
            if 'error' in results:
                continue
                
            html_content += f"""
            <div class="dataset">
                <h3>Датасет: {dataset_name}</h3>
                <p>Файл: {results.get('file_path', 'N/A')}</p>
                <p>Строк: {results.get('rows', 0)} | Колонок: {results.get('columns', 0)}</p>
                
                <h4>Результаты валидации:</h4>
                <table>
                    <tr><th>Проверка</th><th>Результат</th><th>Детали</th></tr>
            """
            
            # Добавляем результаты проверок
            checks = [
                ('Структура', 'structure_checks', results.get('structure_checks', {})),
                ('Пропущенные значения', 'missing_values', results.get('missing_values', {})),
                ('Целевая переменная', 'target_checks', results.get('target_checks', {})),
            ]
            
            for check_name, check_key, check_data in checks:
                if check_data:
                    status = "" if check_key != 'missing_values' or check_data.get('max_missing_percent', 100) <= 5 else ""
                    details = str(check_data)[:100] + "..." if len(str(check_data)) > 100 else str(check_data)
                    html_content += f"<tr><td>{check_name}</td><td>{status}</td><td>{details}</td></tr>"
            
            html_content += f"""
                </table>
                
                <h4>Сводка:</h4>
                <p>Пройдено проверок: {results.get('summary', {}).get('passed_checks', 0)} / 
                   {results.get('summary', {}).get('total_checks', 0)}</p>
                <p>Оценка: {results.get('summary', {}).get('validation_score', 0):.1f}%</p>
                <p>Статус: {' ВАЛИДНО' if results.get('summary', {}).get('is_valid', False) else ' НЕВАЛИДНО'}</p>
            </div>
            """
        
        # Добавляем ошибки и предупреждения
        if report_data['errors']:
            html_content += "<div class='error'><h3>Ошибки:</h3><ul>"
            for error in report_data['errors']:
                html_content += f"<li>{error}</li>"
            html_content += "</ul></div>"
        
        if report_data['warnings']:
            html_content += "<div class='warning'><h3>Предупреждения:</h3><ul>"
            for warning in report_data['warnings']:
                html_content += f"<li>{warning}</li>"
            html_content += "</ul></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_text_report(self, output_path: Path, report_data: Dict):
        """Генерация текстового отчета"""
        with open(output_path, 'w', encoding='utf-8') as f:
           
            f.write(f"Дата: {report_data['timestamp']}\n")
            f.write(f"Статус: {report_data['summary']['overall_status']}\n")
            f.write(f"Ошибки: {report_data['summary']['total_errors']}\n")
            f.write(f"Предупреждения: {report_data['summary']['total_warnings']}\n\n")

            for dataset_name, results in report_data['validation_results'].items():
                if 'error' in results:
                    f.write(f"Датасет: {dataset_name} - ОШИБКА: {results['error']}\n\n")
                    continue
                
                f.write(f"Датасет: {dataset_name}\n")
                f.write(f"Файл: {results.get('file_path', 'N/A')}\n")
                f.write(f"Строк: {results.get('rows', 0)} | Колонок: {results.get('columns', 0)}\n")
                
                if 'summary' in results:
                    f.write(f"Пройдено проверок: {results['summary']['passed_checks']}/{results['summary']['total_checks']}\n")
                    f.write(f"Оценка: {results['summary']['validation_score']:.1f}%\n")
                    f.write(f"Статус: {'ВАЛИДНО' if results['summary']['is_valid'] else 'НЕВАЛИДНО'}\n")
                
                f.write("\n")
            
            if report_data['errors']:
                for error in report_data['errors']:
                    f.write(f"• {error}\n")
            
            if report_data['warnings']:
                for warning in report_data['warnings']:
                    f.write(f"• {warning}\n")

def main():
    parser = argparse.ArgumentParser(description='Валидация данных кредитного скоринга')
    parser.add_argument('--train-data', default='data/processed/train.csv',
                       help='Путь к тренировочным данным')
    parser.add_argument('--test-data', default='data/processed/test.csv',
                       help='Путь к тестовым данным')
    parser.add_argument('--config', default='configs/validation_config.yaml',
                       help='Файл конфигурации валидации')
    parser.add_argument('--output-dir', default='reports/data_validation',
                       help='Директория для сохранения отчетов')
    parser.add_argument('--validate-only', choices=['train', 'test', 'both'], default='both',
                       help='Какие данные валидировать')
    
    args = parser.parse_args()

    # Инициализация валидатора
    validator = DataValidator(args.config)
    
    # Валидация тренировочных данных
    if args.validate_only in ['train', 'both']:
        print(f"\n1. Валидация тренировочных данных:")
        train_results = validator.validate_dataset(args.train_data, 'train')
    
    # Валидация тестовых данных
    if args.validate_only in ['test', 'both']:
        print(f"\n2. Валидация тестовых данных:")
        test_results = validator.validate_dataset(args.test_data, 'test')
    
    # Сохранение результатов
    json_report, html_report = validator.save_results(args.output_dir)
    
    # Вывод сводки

    
    total_errors = len(validator.errors)
    total_warnings = len(validator.warnings)
    
    if total_errors == 0:
        print(" Валидация пройдена успешно")
    else:
        print(f" Обнаружено ошибок: {total_errors}")
        for error in validator.errors[:3]:  # Первые 3 ошибки
            print(f"   - {error}")
        if total_errors > 3:
            print(f"   ... и еще {total_errors - 3} ошибок")
    
    if total_warnings > 0:
        print(f"  Предупреждений: {total_warnings}")
        for warning in validator.warnings[:3]:  # Первые 3 предупреждения
            print(f"   - {warning}")
        if total_warnings > 3:
            print(f"   ... и еще {total_warnings - 3} предупреждений")
        
    return 0 if total_errors == 0 else 1

if __name__ == "__main__":
    exit(main())