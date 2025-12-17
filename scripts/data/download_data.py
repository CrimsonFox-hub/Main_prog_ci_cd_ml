"""
Скрипт для загрузки German Credit Dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import os

def download_german_credit(output_dir: str = "data/raw") -> pd.DataFrame:
    """
    Загрузка German Credit Dataset с UCI Machine Learning Repository
    """
    # URL датасета
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # Названия колонок согласно документации
    column_names = [
        'checking_status', 'duration', 'credit_history', 'purpose', 
        'credit_amount', 'savings_status', 'employment', 'installment_commitment',
        'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
        'age', 'other_payment_plans', 'housing', 'existing_credits', 'job',
        'num_dependents', 'own_telephone', 'foreign_worker', 'class'
    ]
    
    print(f"Загрузка данных с {url}")
    
    try:
        # Загрузка данных
        df = pd.read_csv(url, sep=' ', header=None, names=column_names)
        
        # Преобразование целевой переменной (1 - хороший, 2 - плохой -> 0, 1)
        df['class'] = df['class'].map({1: 0, 2: 1})
        
        # Переименуем целевую переменную для удобства
        df = df.rename(columns={'class': 'default'})
        
        # Создаем директорию если не существует
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Сохраняем данные
        output_path = Path(output_dir) / "german_credit.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Данные сохранены в: {output_path}")
        print(f"Размер датасета: {df.shape}")
        print(f"Колонки: {', '.join(df.columns.tolist())}")
        print(f"Распределение классов:")
        print(df['default'].value_counts())
        print(f"Соотношение классов:")
        print(df['default'].value_counts(normalize=True))
        
        return df
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ЗАМЕНИТЕ функцию create_sample_data в download_data.py:

def create_sample_data(output_dir: str = "data/raw") -> pd.DataFrame:
    """Создание тестовых данных если не удается скачать"""
    print("Создание тестовых данных...")
    
    # Генерация синтетических данных
    np.random.seed(42)
    n_samples = 1000
    
    # Создаем словарь с данными
    data_dict = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_amount': np.random.randint(1000, 50000, n_samples),
        'loan_duration': np.random.randint(6, 60, n_samples),
        'payment_to_income': np.random.uniform(0.1, 0.5, n_samples),
        'existing_credits': np.random.randint(0, 5, n_samples),
        'dependents': np.random.randint(0, 5, n_samples),
        'residence_since': np.random.randint(0, 20, n_samples),
        'installment_rate': np.random.uniform(1.0, 4.0, n_samples),
    }
    
    # Добавляем категориальные признаки как массивы строк
    credit_history = np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples)
    purpose = np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44'], n_samples)
    savings = np.random.choice(['A61', 'A62', 'A63', 'A64'], n_samples)
    
    # Создаем DataFrame
    df = pd.DataFrame(data_dict)
    df['credit_history'] = credit_history
    df['purpose'] = purpose
    df['savings'] = savings
    
    # Простая логика для дефолта
    # Используем DataFrame для работы с .isin
    default_condition = (
        (df['payment_to_income'] > 0.4).astype(int) |
        (df['credit_history'].isin(['A33', 'A34'])).astype(int) |
        (df['age'] < 25).astype(int)
    )
    
    df['default'] = default_condition
    
    # Создаем директорию если не существует
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Сохраняем данные
    output_path = Path(output_dir) / "credit_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Тестовые данные сохранены в: {output_path}")
    print(f"Размер датасета: {df.shape}")
    print(f"Распределение классов:")
    print(df['default'].value_counts())
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Загрузка кредитных данных')
    parser.add_argument('--output-dir', default='data/raw', help='Директория для сохранения данных')
    parser.add_argument('--use-sample', action='store_true', help='Использовать тестовые данные вместо загрузки')
    parser.add_argument('--source-url', help='URL для загрузки данных')
    
    args = parser.parse_args()
    
    if args.use_sample:
        df = create_sample_data(args.output_dir)
    else:
        try:
            df = download_german_credit(args.output_dir)
        except Exception as e:
            print(f"Не удалось загрузить данные: {e}")
            print("Создаем тестовые данные...")
            df = create_sample_data(args.output_dir)
    
    # Создание дополнительных файлов для DVC
    from sklearn.model_selection import train_test_split
    
    # Разделение на train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['default'])
    
    # Сохраняем разделенные данные
    processed_dir = Path(args.output_dir).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nДанные разделены и сохранены:")
    print(f"Train: {train_path} ({len(train_df)} samples)")
    print(f"Test: {test_path} ({len(test_df)} samples)")
    
    # Создаем файлы для DVC
    dvc_dir = Path(".dvc") / "data"
    dvc_dir.mkdir(parents=True, exist_ok=True)
    
    with open(dvc_dir / "data_info.json", "w") as f:
        import json
        json.dump({
            "dataset": "German Credit" if not args.use_sample else "Sample Credit Data",
            "samples": len(df),
            "features": len(df.columns) - 1,
            "positive_class_ratio": df['default'].mean(),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "download_date": pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    print("\nЗагрузка данных завершена!")

if __name__ == "__main__":
    main()