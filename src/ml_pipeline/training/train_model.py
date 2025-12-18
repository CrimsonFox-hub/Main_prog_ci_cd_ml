"""
Обучение модели для кредитного скоринга
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import yaml
import os
from pathlib import Path

def load_data(config):
    """Загрузка данных"""
    print("Загрузка данных...")
    
    train_df = pd.read_csv(config['data']['train_path'])
    test_df = pd.read_csv(config['data']['test_path'])
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

def train_model(config):
    """Основная функция обучения"""
    print("Начало обучения модели...")
    
    # Загрузка данных
    train_df, test_df = load_data(config)
    
    # Разделение на признаки и целевую переменную
    target_column = config['data']['target_column']
    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    print(f"Признаки: {X_train.shape[1]}, Образцы: {X_train.shape[0]}")
    print(f"Классы: {dict(y_train.value_counts())}")
    
    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Оценка модели
    print("\nМетрики модели:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Сохранение модели
    model_path = config.get('model_paths', {}).get('trained', 'models/trained/model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    # Сохранение метрик
    metrics = {
        'accuracy': float((y_pred == y_test).mean()),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }
    
    metrics_path = 'models/trained/metrics.json'
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nМодель сохранена: {model_path}")
    print(f"Метрики сохранены: {metrics_path}")
    
    return model

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение модели кредитного скоринга')
    parser.add_argument('--config', default='configs/training_config.yaml', 
                       help='Путь к конфигурационному файлу')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    print(f"Загрузка конфигурации из {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Проверка и создание директорий
    if 'model_paths' in config:
        for key, path in config['model_paths'].items():
            if isinstance(path, str):
                os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Обучение модели
    model = train_model(config)
    print("\nОбучение завершено успешно!")

if __name__ == "__main__":
    main()