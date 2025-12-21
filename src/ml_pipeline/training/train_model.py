"""
Обучение нейронной сети для кредитного скоринга
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
import yaml
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import train_logger
from src.utils.config_loader import get_config

# Определение архитектуры нейронной сети
class CreditScoringNN(nn.Module):
    """Нейронная сеть для кредитного скоринга"""
    
    def __init__(self, input_size, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        super(CreditScoringNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Динамическое создание слоев
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Выходной слой
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Dataset для PyTorch
class CreditDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_prepare_data(config_path='configs/training_config.yaml'):
    """Загрузка и подготовка данных"""
    config = get_config('training', config_path)
    
    # Загрузка данных
    data_path = Path(config['data_path'])
    train_logger.info(f"Loading data from: {data_path}")
    
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    train_logger.info(f"Data shape: {df.shape}")
    train_logger.info(f"Columns: {df.columns.tolist()}")
    
    # Определение признаков и целевой переменной
    target_column = config['target_column']
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    # Разделение на train/test
    test_size = config.get('test_size', 0.2)
    random_state = config.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Препроцессинг
    numerical_features = config.get('numerical_features', [])
    categorical_features = config.get('categorical_features', [])
    
    # Проверка наличия признаков в данных
    numerical_features = [f for f in numerical_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Создание пайплайна препроцессинга
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Обучение препроцессора на тренировочных данных
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Сохранение препроцессора
    preprocessor_path = Path(config.get('preprocessor_path', 'models/preprocessor.joblib'))
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    train_logger.info(f"Preprocessor saved to: {preprocessor_path}")
    
    # Расчет итогового размера признаков
    input_size = X_train_processed.shape[1]
    train_logger.info(f"Input size after preprocessing: {input_size}")
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'input_size': input_size,
        'feature_names': numerical_features + categorical_features
    }

def train_model(config_path='configs/training_config.yaml'):
    """Обучение модели"""
    config = get_config('training', config_path)
    
    # Настройка MLflow
    mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
    mlflow.set_experiment(config.get('experiment_name', 'credit_scoring'))
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Загрузка данных
        data = load_and_prepare_data(config_path)
        
        # Создание даталоадеров
        batch_size = config.get('batch_size', 32)
        
        train_dataset = CreditDataset(data['X_train'], data['y_train'])
        test_dataset = CreditDataset(data['X_test'], data['y_test'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Создание модели
        model_config = config.get('model_config', {})
        model = CreditScoringNN(
            input_size=data['input_size'],
            hidden_layers=model_config.get('hidden_layers', [128, 64, 32]),
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )
        
        # Настройка обучения
        learning_rate = config.get('learning_rate', 0.001)
        num_epochs = config.get('num_epochs', 100)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Логирование параметров в MLflow
        mlflow.log_params({
            'input_size': data['input_size'],
            'hidden_layers': model_config.get('hidden_layers', [128, 64, 32]),
            'dropout_rate': model_config.get('dropout_rate', 0.3),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        })
        
        # Обучение
        train_logger.info("Starting training...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        best_loss = float('inf')
        patience = config.get('early_stopping_patience', 10)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Тренировочная эпоха
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_dataset)
            
            # Валидационная эпоха
            model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                    
                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == batch_labels).sum().item()
            
            val_loss /= len(test_loader)
            val_acc = val_correct / len(test_dataset)
            
            # Обновление scheduler
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Сохранение лучшей модели
                model_path = Path(config.get('model_save_path', 'models/credit_scoring.pth'))
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                train_logger.info(f"Best model saved to: {model_path}")
            else:
                patience_counter += 1
            
            # Логирование метрик
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            if epoch % 10 == 0:
                train_logger.info(
                    f"Epoch {epoch}/{num_epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            
            if patience_counter >= patience:
                train_logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Загрузка лучшей модели для оценки
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Оценка модели
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        with torch.no_grad():
            test_features = torch.FloatTensor(data['X_test']).to(device)
            predictions = model(test_features).cpu().numpy().squeeze()
            predictions_binary = (predictions > 0.5).astype(int)
        
        # Расчет метрик
        metrics = {
            'accuracy': accuracy_score(data['y_test'], predictions_binary),
            'precision': precision_score(data['y_test'], predictions_binary, zero_division=0),
            'recall': recall_score(data['y_test'], predictions_binary, zero_division=0),
            'f1_score': f1_score(data['y_test'], predictions_binary, zero_division=0),
            'roc_auc': roc_auc_score(data['y_test'], predictions)
        }
        
        train_logger.info(f"Test Metrics: {metrics}")
        
        # Логирование метрик в MLflow
        mlflow.log_metrics(metrics)
        
        # Сохранение модели в MLflow
        mlflow.pytorch.log_model(model, "credit_scoring_model")
        
        # Сохранение метрик в файл
        metrics_path = Path('models/training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        train_logger.info(f"Metrics saved to: {metrics_path}")
        
        return {
            'model': model,
            'metrics': metrics,
            'preprocessor': data['preprocessor'],
            'input_size': data['input_size'],
            'model_path': str(model_path)
        }

def evaluate_model(model, test_loader, device):
    """Оценка модели"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features).squeeze()
            predictions = (outputs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

if __name__ == "__main__":
    # Пример использования
    results = train_model()
    train_logger.info(f"Training completed successfully!")
    train_logger.info(f"Model saved to: {results['model_path']}")
    train_logger.info(f"Test metrics: {results['metrics']}")