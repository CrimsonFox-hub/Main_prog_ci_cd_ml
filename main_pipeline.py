#!/usr/bin/env python3
"""
Основной пайплайн для обучения нейронной сети кредитного скоринга
Этап 1: Подготовка модели к промышленной эксплуатации
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json
import yaml
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import mlflow
import mlflow.pytorch

# Импорт наших модулей
from src.ml_pipeline.training.neural_network import CreditScoringNN
from src.ml_pipeline.training.onnx_conversion import ModelConverter
from src.ml_pipeline.optimization.model_optimizer import ModelOptimizer
from src.ml_pipeline.optimization.benchmark import ModelBenchmark
from src.utils.logger import setup_logger

# Настройка логирования
logger = setup_logger('training')

def load_and_prepare_data(config_path='configs/training_config.yaml'):
    """Загрузка и подготовка данных"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loading data from: {config['data_path']}")
    
    # Загрузка данных
    df = pd.read_csv(config['data_path'])
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Проверка целевой переменной
    target_column = config['target_column']
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Разделение на признаки и целевую переменную
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    # Разделение на train/test
    test_size = config.get('test_size', 0.2)
    random_state = config.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Препроцессинг числовых признаков
    numerical_features = config.get('numerical_features', [])
    categorical_features = config.get('categorical_features', [])
    
    # Фильтрация существующих признаков
    numerical_features = [f for f in numerical_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Масштабирование числовых признаков
    scaler = StandardScaler()
    X_train_numerical = scaler.fit_transform(X_train[numerical_features])
    X_test_numerical = scaler.transform(X_test[numerical_features])
    
    # Кодирование категориальных признаков
    encoders = {}
    X_train_categorical = []
    X_test_categorical = []
    
    for feature in categorical_features:
        encoder = LabelEncoder()
        X_train_cat = encoder.fit_transform(X_train[feature]).reshape(-1, 1)
        X_test_cat = encoder.transform(X_test[feature]).reshape(-1, 1)
        
        X_train_categorical.append(X_train_cat)
        X_test_categorical.append(X_test_cat)
        encoders[feature] = encoder
    
    # Объединение признаков
    if X_train_categorical:
        X_train_categorical = np.hstack(X_train_categorical)
        X_test_categorical = np.hstack(X_test_categorical)
        
        X_train_processed = np.hstack([X_train_numerical, X_train_categorical])
        X_test_processed = np.hstack([X_test_numerical, X_test_categorical])
    else:
        X_train_processed = X_train_numerical
        X_test_processed = X_test_numerical
    
    # Сохранение препроцессоров
    preprocessors = {
        'scaler': scaler,
        'encoders': encoders,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    
    preprocessor_path = config.get('preprocessor_path', 'models/preprocessors.joblib')
    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessors, preprocessor_path)
    logger.info(f"Preprocessors saved to: {preprocessor_path}")
    
    # Расчет размера входных данных
    input_size = X_train_processed.shape[1]
    logger.info(f"Input size: {input_size}")
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessors': preprocessors,
        'input_size': input_size,
        'feature_names': numerical_features + categorical_features
    }

def train_neural_network(data, config_path='configs/training_config.yaml'):
    """Обучение нейронной сети"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Настройка MLflow
    mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
    mlflow.set_experiment(config.get('experiment_name', 'credit_scoring_nn'))
    
    with mlflow.start_run(run_name=f"nn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Конфигурация модели
        model_config = config.get('model_config', {})
        model_type = model_config.get('type', 'simple')
        hidden_layers = model_config.get('hidden_layers', [128, 64, 32])
        dropout_rate = model_config.get('dropout_rate', 0.3)
        
        # Создание модели
        if model_type == 'simple':
            from src.ml_pipeline.training.neural_network import CreditScoringNN
            model = CreditScoringNN(
                input_size=data['input_size'],
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
        # Параметры обучения
        training_config = config.get('training', {})
        batch_size = training_config.get('batch_size', 32)
        learning_rate = training_config.get('learning_rate', 0.001)
        num_epochs = training_config.get('num_epochs', 100)
        early_stopping_patience = training_config.get('early_stopping_patience', 10)
        
        # Подготовка данных для PyTorch
        X_train_tensor = torch.FloatTensor(data['X_train'])
        y_train_tensor = torch.FloatTensor(data['y_train']).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(data['X_test'])
        y_test_tensor = torch.FloatTensor(data['y_test']).reshape(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Настройка оптимизатора и функции потерь
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Логирование параметров в MLflow
        mlflow.log_params({
            'model_type': model_type,
            'input_size': data['input_size'],
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs
        })
        
        # Обучение
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"Training on device: {device}")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Тренировочная эпоха
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_dataset)
            
            # Валидационная эпоха
            model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
            
            val_loss /= len(test_loader)
            val_acc = val_correct / len(test_dataset)
            
            # Обновление scheduler
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Сохранение лучшей модели
                model_path = config.get('model_save_path', 'models/credit_scoring_nn.pth')
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                logger.info(f"Best model saved to: {model_path}")
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
                logger.info(
                    f"Epoch {epoch}/{num_epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Загрузка лучшей модели
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Оценка на тестовых данных
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device)).cpu().numpy()
            test_predictions = (test_outputs > 0.5).astype(int)
        
        # Расчет метрик
        metrics = {
            'accuracy': accuracy_score(data['y_test'], test_predictions),
            'precision': precision_score(data['y_test'], test_predictions, zero_division=0),
            'recall': recall_score(data['y_test'], test_predictions, zero_division=0),
            'f1_score': f1_score(data['y_test'], test_predictions, zero_division=0),
            'roc_auc': roc_auc_score(data['y_test'], test_outputs),
            'confusion_matrix': confusion_matrix(data['y_test'], test_predictions).tolist()
        }
        
        logger.info(f"Test Metrics: {metrics}")
        
        # Логирование в MLflow
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "credit_scoring_nn")
        
        # Сохранение метрик
        metrics_path = 'models/training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)
        
        logger.info(f"Metrics saved to: {metrics_path}")
        
        return {
            'model': model,
            'metrics': metrics,
            'model_path': model_path,
            'input_size': data['input_size']
        }

def convert_to_onnx(model, input_size, model_path):
    """Конвертация модели в ONNX"""
    logger.info("Converting model to ONNX...")
    
    converter = ModelConverter(model_path, input_size)
    onnx_path = converter.convert_to_onnx()
    
    # Валидация конвертации
    validation_results = converter.validate_conversion(onnx_path)
    logger.info(f"Validation results: {validation_results}")
    
    return onnx_path

def optimize_model(model, input_size, model_path):
    """Оптимизация модели"""
    logger.info("Optimizing model...")
    
    optimizer = ModelOptimizer(model_path, input_size)
    
    # Pruning
    pruned_model = optimizer.apply_pruning(model, pruning_rate=0.2)
    pruned_path = model_path.replace('.pth', '_pruned.pth')
    torch.save(pruned_model.state_dict(), pruned_path)
    logger.info(f"Pruned model saved to: {pruned_path}")
    
    # Quantization
    quantized_model = optimizer.dynamic_quantization(pruned_model)
    quantized_path = model_path.replace('.pth', '_quantized.pth')
    torch.jit.save(torch.jit.script(quantized_model), quantized_path)
    logger.info(f"Quantized model saved to: {quantized_path}")
    
    return pruned_path, quantized_path

def run_benchmark(model_paths):
    """Запуск бенчмарка моделей"""
    logger.info("Running benchmark...")
    
    benchmark = ModelBenchmark(model_paths)
    results = benchmark.benchmark_models()
    
    # Сохранение результатов
    results_path = 'reports/benchmark_results.json'
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to: {results_path}")
    
    return results

def main():
    """Основной пайплайн"""
    logger.info("=" * 80)
    logger.info("CREDIT SCORING NEURAL NETWORK PIPELINE")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Загрузка и подготовка данных
        logger.info("\n1. Loading and preparing data...")
        data = load_and_prepare_data()
        
        # 2. Обучение нейронной сети
        logger.info("\n2. Training neural network...")
        training_results = train_neural_network(data)
        
        # 3. Конвертация в ONNX
        logger.info("\n3. Converting to ONNX...")
        onnx_path = convert_to_onnx(
            training_results['model'],
            training_results['input_size'],
            training_results['model_path']
        )
        
        # 4. Оптимизация модели
        logger.info("\n4. Optimizing model...")
        pruned_path, quantized_path = optimize_model(
            training_results['model'],
            training_results['input_size'],
            training_results['model_path']
        )
        
        # 5. Бенчмаркинг
        logger.info("\n5. Running benchmark...")
        model_paths = {
            'original_pytorch': training_results['model_path'],
            'pruned_pytorch': pruned_path,
            'quantized_pytorch': quantized_path,
            'onnx': onnx_path
        }
        
        benchmark_results = run_benchmark(model_paths)
        
        # 6. Создание отчета
        logger.info("\n6. Generating report...")
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_metrics': training_results['metrics'],
            'onnx_validation': {'path': onnx_path},
            'optimization': {
                'pruned_model': pruned_path,
                'quantized_model': quantized_path
            },
            'benchmark_summary': benchmark_results.get('comparison', {}),
            'input_size': training_results['input_size'],
            'total_time_minutes': (time.time() - start_time) / 60
        }
        
        report_path = 'reports/pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nPipeline completed successfully!")
        logger.info(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")
        logger.info(f"   Model metrics: {training_results['metrics']}")
        logger.info(f"   Report saved to: {report_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()