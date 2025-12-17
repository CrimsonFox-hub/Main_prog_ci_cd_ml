"""
Упрощенный скрипт обучения для тестирования
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class SimpleNN(nn.Module):
    """Простая нейронная сеть"""
    def __init__(self, input_size, hidden_layers=[64, 32], dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return self.sigmoid(x)

# ЗАМЕНИТЕ функцию load_and_preprocess_data в train_model_simple.py:

def load_and_preprocess_data(config):
    """Загрузка и предобработка данных"""
    print("Загрузка данных...")
    
    # Пробуем загрузить обработанные данные
    train_path = Path(config['data']['train_path'])
    
    if train_path.exists():
        df = pd.read_csv(train_path)
        print(f"Загружено {len(df)} образцов из {train_path}")
    else:
        # Создаем тестовые данные
        print("Создание тестовых данных...")
        np.random.seed(42)
        n_samples = 800
        
        # ТОЛЬКО числовые данные для упрощения
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
            'default': np.random.binomial(1, 0.3, n_samples)
        }
        
        df = pd.DataFrame(data)
        print(f"Создано {len(df)} тестовых образцов")
    
    # Разделение на признаки и целевую переменную
    # Берем только числовые признаки
    numerical_features = [
        'age', 'income', 'credit_amount', 'loan_duration', 'payment_to_income',
        'existing_credits', 'dependents', 'residence_since', 'installment_rate'
    ]
    
    # Оставляем только существующие колонки
    numerical_features = [col for col in numerical_features if col in df.columns]
    
    X = df[numerical_features]
    y = df[config['data']['target_column']]
    
    print(f"Используем {len(numerical_features)} числовых признаков: {numerical_features}")
    
    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Сохраняем скейлер
    scaler_path = Path(config['model']['paths']['scaler'])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Скейлер сохранен в {scaler_path}")
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (torch.FloatTensor(X_train), torch.FloatTensor(y_train.values),
            torch.FloatTensor(X_val), torch.FloatTensor(y_val.values))

def train_simple_model(config):
    """Обучение простой модели"""
    print("Начало обучения...")
    
    # Загрузка данных
    X_train, y_train, X_val, y_val = load_and_preprocess_data(config)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # Создание модели
    input_size = X_train.shape[1]
    model = SimpleNN(
        input_size=input_size,
        hidden_layers=config['model']['hidden_layers'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Функция потерь и оптимизатор
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Обучение
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val).item()
            val_losses.append(val_loss)
            
            # Accuracy
            predictions = (val_outputs.squeeze() > 0.5).float()
            accuracy = (predictions == y_val).float().mean().item()
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(config['model']['paths']['best_model'])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model to {model_path}")
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}")
    
    # Сохранение финальной модели
    final_model_path = Path(config['model']['paths']['final_model'])
    torch.save(model.state_dict(), final_model_path)
    print(f"Финальная модель сохранена в {final_model_path}")
    
    # Сохранение метрик
    metrics = {
        'best_val_loss': best_val_loss,
        'final_accuracy': accuracy,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_epochs': config['training']['epochs'],
        'input_size': input_size
    }
    
    metrics_path = Path(config['model']['paths']['metrics'])
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Метрики сохранены в {metrics_path}")
    print(f"Обучение завершено. Best val loss: {best_val_loss:.4f}")
    
    return model

def main():
    """Основная функция"""
    print("=== Упрощенное обучение модели ===")
    
    # Загрузка конфигурации
    config_path = "configs/training_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Конфигурация загружена из {config_path}")
        print(f"Проект: {config['project']['name']} v{config['project']['version']}")
        
        # Создаем необходимые директории
        for path_key in ['best_model', 'final_model', 'onnx_model', 'scaler', 'metrics']:
            if path_key in config['model']['paths']:
                path = Path(config['model']['paths'][path_key])
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Создана директория: {path.parent}")
        
        # Обучаем модель
        model = train_simple_model(config)
        
        # Простой тест модели
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 20)  # 20 признаков
            prediction = model(test_input)
            print(f"\nТестовый инференс: {prediction.item():.4f}")
        
        print("\n=== Обучение успешно завершено ===")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()