"""
Простой скрипт для создания тестовых данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_sample_data():
    """Создание простых тестовых данных"""
    print("Создание тестовых данных...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Создаем только числовые данные для упрощения
    data = {
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
    
    # Простая логика для дефолта
    data['default'] = (
        (data['payment_to_income'] > 0.4).astype(int) |
        (data['age'] < 25).astype(int) |
        (np.random.random(n_samples) > 0.7).astype(int)
    )
    
    df = pd.DataFrame(data)
    
    # Создаем директории
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем сырые данные
    raw_path = raw_dir / "credit_data.csv"
    df.to_csv(raw_path, index=False)
    print(f"Сырые данные сохранены в: {raw_path}")
    
    # Разделяем на train/test
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['default']
    )
    
    # Сохраняем обработанные данные
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train данные сохранены в: {train_path} ({len(train_df)} samples)")
    print(f"Test данные сохранены в: {test_path} ({len(test_df)} samples)")
    
    print(f"\nРаспределение классов:")
    print(f"Train: {train_df['default'].mean():.2%} дефолтов")
    print(f"Test: {test_df['default'].mean():.2%} дефолтов")
    
    print(f"\nСтруктура данных:")
    print(f"Колонки: {', '.join(df.columns.tolist())}")
    print(f"Типы данных:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    return df, train_df, test_df

if __name__ == "__main__":
    create_sample_data()
    print("\n✅ Тестовые данные успешно созданы!")