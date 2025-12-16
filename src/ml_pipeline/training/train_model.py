"""
Обучение нейронной сети для кредитного скоринга
Этап 1: Подготовка модели к промышленной эксплуатации
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.pytorch
import yaml
import json
from datetime import datetime
from pathlib import Path

class CreditDataset(Dataset):
    """Датасет для кредитных данных"""
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class CreditScoringNN(nn.Module):
    """Нейронная сеть для кредитного скоринга"""
    def __init__(self, input_size, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        super(CreditScoringNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

def load_and_preprocess_data(config):
    """Загрузка и предобработка данных"""
    print("Loading data...")
    
    # Загрузка данных из DVC
    train_df = pd.read_csv(config['data_paths']['train'])
    test_df = pd.read_csv(config['data_paths']['test'])
    
    # Разделение на признаки и целевую переменную
    X_train = train_df.drop(columns=[config['target_column']])
    y_train = train_df[config['target_column']]
    X_test = test_df.drop(columns=[config['target_column']])
    y_test = test_df[config['target_column']]
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Сохранение скейлера
    joblib.dump(scaler, config['model_paths']['scaler'])
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

def train_model(config):
    """Основная функция обучения"""
    # Настройка MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_params(config['training'])
        mlflow.log_param("hidden_layers", config['model']['hidden_layers'])
        
        # Загрузка данных
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(config)
        
        # Создание датасетов
        train_dataset = CreditDataset(X_train, y_train)
        test_dataset = CreditDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size']
        )
        
        # Инициализация модели
        input_size = X_train.shape[1]
        model = CreditScoringNN(
            input_size=input_size,
            hidden_layers=config['model']['hidden_layers'],
            dropout_rate=config['model']['dropout_rate']
        )
        
        # Функция потерь и оптимизатор
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Обучение
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(config['training']['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
                    
                    # Вычисление точности
                    predictions = (outputs.squeeze() > 0.5).float()
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
            
            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)
            accuracy = correct / total
            
            # Логирование метрик
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            
            # Сохранение лучшей модели
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config['model_paths']['best_model'])
                mlflow.pytorch.log_model(model, "best_model")
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Сохранение финальной модели
        torch.save(model.state_dict(), config['model_paths']['final_model'])
        
        # Сохранение метрик
        metrics = {
            'best_val_loss': best_val_loss,
            'final_accuracy': accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        with open(config['model_paths']['metrics'], 'w') as f:
            json.dump(metrics, f)
        
        mlflow.log_artifact(config['model_paths']['metrics'])
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return model

if __name__ == "__main__":
    # Загрузка конфигурации
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Создание директорий
    for path in config['model_paths'].values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Обучение модели
    model = train_model(config)