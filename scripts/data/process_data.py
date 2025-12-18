"""
Скрипт для обработки кредитных данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import yaml
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(input_path: str) -> pd.DataFrame:
    """Загрузка данных из CSV файла"""
    print(f"Загрузка данных из {input_path}")
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
    
    return df

def preprocess_data(df: pd.DataFrame, config: dict, fit: bool = True) -> pd.DataFrame:
    """Предобработка данных"""
    print("Предобработка данных...")
    
    # Определяем числовые и категориальные признаки
    numerical_features = config['data']['numerical_features']
    categorical_features = config['data']['categorical_features']
    target_column = config['data']['target_column']
    
    # Проверяем наличие всех колонок
    missing_numerical = [col for col in numerical_features if col not in df.columns]
    missing_categorical = [col for col in categorical_features if col not in df.columns]
    
    if missing_numerical:
        print(f"Предупреждение: Отсутствуют числовые признаки: {missing_numerical}")
    
    if missing_categorical:
        print(f"Предупреждение: Отсутствуют категориальные признаки: {missing_categorical}")
    
    # Оставляем только существующие признаки
    numerical_features = [col for col in numerical_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    # Создаем трансформер
    if fit:
        transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'
        )
        
        # Сохраняем трансформер
        Path("models/trained").mkdir(parents=True, exist_ok=True)
        joblib.dump(transformer, config['model']['paths']['scaler'])
        print(f"Трансформер сохранен в {config['model']['paths']['scaler']}")
    else:
        # Загружаем сохраненный трансформер
        transformer = joblib.load(config['model']['paths']['scaler'])
    
    # Применяем трансформации
    features = transformer.fit_transform(df) if fit else transformer.transform(df)
    
    # Создаем DataFrame с трансформированными признаками
    feature_names = []
    
    # Имена числовых признаков
    feature_names.extend(numerical_features)
    
    # Имена категориальных признаков (после one-hot кодирования)
    if categorical_features and hasattr(transformer.named_transformers_['cat'], 'get_feature_names_out'):
        cat_features = transformer.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_features)
    
    # Создаем DataFrame
    processed_df = pd.DataFrame(features, columns=feature_names)
    
    # Добавляем целевую переменную если она есть
    if target_column in df.columns:
        processed_df[target_column] = df[target_column].values
    
    print(f"После обработки: {processed_df.shape[0]} строк, {processed_df.shape[1]} колонок")
    
    return processed_df

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> tuple:
    """Разделение данных на train/test"""
    print(f"Разделение данных (test_size={test_size})...")
    
    from sklearn.model_selection import train_test_split
    
    # Разделяем на признаки и целевую переменную
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Объединяем обратно в DataFrame
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    print(f"Train: {len(train_df)} samples, {train_df[target_column].mean():.2%} positive")
    print(f"Test: {len(test_df)} samples, {test_df[target_column].mean():.2%} positive")
    
    return train_df, test_df

def create_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Создание новых признаков"""
    print("Создание новых признаков...")
    
    # Копируем DataFrame
    df_eng = df.copy()
    
    # 1. Отношение кредита к доходу
    if 'credit_amount' in df_eng.columns and 'income' in df_eng.columns:
        df_eng['debt_to_income_ratio'] = df_eng['credit_amount'] / df_eng['income']
    
    # 2. Ежемесячный платеж
    if 'credit_amount' in df_eng.columns and 'loan_duration' in df_eng.columns:
        df_eng['monthly_payment'] = df_eng['credit_amount'] / df_eng['loan_duration']
    
    # 3. Возрастные группы
    if 'age' in df_eng.columns:
        df_eng['age_group'] = pd.cut(
            df_eng['age'],
            bins=[18, 25, 35, 50, 65, 100],
            labels=['18-25', '26-35', '36-50', '51-65', '66+']
        )
    
    # 4. Отношение платежа к доходу (уже есть как payment_to_income)
    
    print(f"Добавлено {len(df_eng.columns) - len(df.columns)} новых признаков")
    
    return df_eng

def main():
    parser = argparse.ArgumentParser(description='Обработка кредитных данных')
    parser.add_argument('--input', default='data/raw/credit_data.csv', help='Входной файл с данными')
    parser.add_argument('--output-dir', default='data/processed', help='Директория для сохранения обработанных данных')
    parser.add_argument('--config', default='configs/training_config.yaml', help='Файл конфигурации')
    parser.add_argument('--feature-engineering', action='store_true', help='Создание новых признаков')
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Загружаем данные
        df = load_data(args.input)
        
        # Создаем новые признаки если нужно
        if args.feature_engineering:
            df = create_feature_engineering(df)
        
        # Предобработка данных
        processed_df = preprocess_data(df, config, fit=True)
        
        # Разделение на train/test если есть целевая переменная
        target_column = config['data']['target_column']
        
        if target_column in processed_df.columns:
            train_df, test_df = split_data(processed_df, target_column)
            
            # Создаем директорию если не существует
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Сохраняем данные
            train_path = Path(args.output_dir) / "train.csv"
            test_path = Path(args.output_dir) / "test.csv"
            processed_path = Path(args.output_dir) / "processed.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            processed_df.to_csv(processed_path, index=False)
            
            print(f"\nДанные сохранены:")
            print(f"  Train: {train_path} ({len(train_df)} samples)")
            print(f"  Test: {test_path} ({len(test_df)} samples)")
            print(f"  Processed: {processed_path} ({len(processed_df)} samples)")
            
            # Сохраняем метаданные
            metadata = {
                "original_samples": len(df),
                "processed_samples": len(processed_df),
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "features": len(processed_df.columns) - 1,
                "numerical_features": len(config['data']['numerical_features']),
                "categorical_features": len(config['data']['categorical_features']),
                "target_distribution": {
                    "train": {
                        "positive": int(train_df[target_column].sum()),
                        "negative": int(len(train_df) - train_df[target_column].sum()),
                        "ratio": float(train_df[target_column].mean())
                    },
                    "test": {
                        "positive": int(test_df[target_column].sum()),
                        "negative": int(len(test_df) - test_df[target_column].sum()),
                        "ratio": float(test_df[target_column].mean())
                    }
                },
                "processing_date": pd.Timestamp.now().isoformat()
            }
            
            metadata_path = Path(args.output_dir) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Metadata: {metadata_path}")
        
        print("\nОбработка данных завершена успешно!")
        
    except Exception as e:
        print(f"Ошибка при обработке данных: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()