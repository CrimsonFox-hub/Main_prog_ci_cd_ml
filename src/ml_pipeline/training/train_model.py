"""
Промышленный пайплайн обучения модели с ONNX экспортом
Этап 1: Подготовка модели к промышленной эксплуатации
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
import onnx
import onnxruntime as ort
from pathlib import Path
import json
import time
from datetime import datetime
import yaml
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
warnings.filterwarnings('ignore')

# ==================== 1. НЕЙРОННАЯ СЕТЬ ====================
class CreditScoringNN(nn.Module):
    """Нейронная сеть для кредитного скоринга"""
    def __init__(self, input_size, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        super(CreditScoringNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
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

# ==================== 2. ДАТАСЕТ ====================
class CreditDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.has_labels = labels is not None
        if self.has_labels:
            self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.has_labels:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

# ==================== 3. ОБУЧЕНИЕ ====================
def train_neural_network(config):
    """Обучение нейронной сети"""
    print("=" * 60)
    print("🚀 ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    print("=" * 60)
    
    # Загрузка данных
    print("\n📥 Загрузка данных...")
    train_df = pd.read_csv(config['data']['train_path'])
    test_df = pd.read_csv(config['data']['test_path'])
    
    target_col = config['data']['target_column']
    
    X_train = train_df.drop(columns=[target_col]).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    X_test = test_df.drop(columns=[target_col]).values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Создание датасетов
    train_dataset = CreditDataset(X_train, y_train)
    test_dataset = CreditDataset(X_test, y_test)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
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
    epochs = config['training']['epochs']
    print(f"\n⚙️  Настройки:")
    print(f"   Эпохи: {epochs}, Batch: {batch_size}, LR: {config['training']['learning_rate']}")
    print(f"   Архитектура: {config['model']['hidden_layers']}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
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
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
                
                all_preds.extend(outputs.squeeze().numpy())
                all_labels.extend(batch_y.numpy())
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        # Метрики
        predictions = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:3d}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'accuracy': accuracy,
                'f1_score': f1,
                'input_size': input_size
            }, config['model_paths']['best_model'])
    
    # Сохранение финальной модели
    torch.save(model.state_dict(), config['model_paths']['final_model'])
    
    # Тестирование финальной модели
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            outputs = model(batch_x)
            test_preds.extend(outputs.squeeze().numpy())
    
    # Метрики на тесте
    test_predictions = (np.array(test_preds) > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    test_roc_auc = roc_auc_score(y_test, test_preds)
    
    print(f"\n✅ Обучение завершено!")
    print(f"   Лучшая валидационная loss: {best_val_loss:.4f}")
    print(f"   Тест Accuracy: {test_accuracy:.4f}")
    print(f"   Тест F1: {test_f1:.4f}")
    print(f"   Тест ROC-AUC: {test_roc_auc:.4f}")
    
    # Сохранение метрик
    metrics = {
        'best_val_loss': float(best_val_loss),
        'test_accuracy': float(test_accuracy),
        'test_f1_score': float(test_f1),
        'test_roc_auc': float(test_roc_auc),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'input_size': input_size,
        'training_time': datetime.now().isoformat(),
        'model_architecture': str(config['model']['hidden_layers']),
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': config['training']['learning_rate']
        }
    }
    
    with open(config['model_paths']['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📊 Метрики сохранены: {config['model_paths']['metrics']}")
    
    return model, input_size

# ==================== 4. ONNX КОНВЕРТАЦИЯ ====================
def convert_to_onnx(model, input_size, config):
    """Конвертация в ONNX формат"""
    print("\n" + "=" * 60)
    print("🔄 КОНВЕРТАЦИЯ В ONNX")
    print("=" * 60)
    
    model.eval()
    
    # Используем динамический batch size
    batch_size = 1  # для экспорта
    dummy_input = torch.randn(batch_size, input_size, requires_grad=True)
    
    onnx_path = config['model_paths']['onnx']
    
    print(f"   Экспорт модели в ONNX: {onnx_path}")
    
    # Экспорт с динамическими осями
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # динамическая ось batch
            'output': {0: 'batch_size'}
        },
        # Добавляем метаданные
        metadata={'model_type': 'credit_scoring', 'version': '1.0.0'}
    )
    
    # Проверка ONNX модели
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("   ✅ ONNX модель валидна")
        
        # Проверка совместимости с ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        print(f"   ✅ ONNX Runtime сессия создана")
        print(f"   🏷️  Входные имена: {[input.name for input in ort_session.get_inputs()]}")
        print(f"   🏷️  Выходные имена: {[output.name for output in ort_session.get_outputs()]}")
        
    except Exception as e:
        print(f"   ⚠️  Предупреждение при проверке ONNX: {str(e)}")
        print("   Продолжаем выполнение...")
    
    return onnx_path

# ==================== 5. БЕНЧМАРКИНГ ====================
def benchmark_models(config, input_size):
    """Сравнение производительности"""
    print("\n" + "=" * 60)
    print("⚡ БЕНЧМАРКИНГ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    # Подготовка тестовых данных
    dummy_input_np = np.random.randn(1000, input_size).astype(np.float32)
    dummy_input_torch = torch.FloatTensor(dummy_input_np)
    
    results = []
    
    # 1. PyTorch модель
    print("\n1. PyTorch модель (CPU):")
    model = CreditScoringNN(input_size)
    model.load_state_dict(torch.load(config['model_paths']['final_model']))
    model.eval()
    
    start = time.time()
    with torch.no_grad():
        for i in range(0, 1000, 100):
            batch = dummy_input_torch[i:i+100]
            _ = model(batch)
    torch_time = time.time() - start
    
    results.append({
        'model': 'PyTorch',
        'samples_per_sec': 1000 / torch_time,
        'latency_ms': (torch_time / 1000) * 1000
    })
    print(f"   Скорость: {1000/torch_time:.2f} samples/sec")
    print(f"   Задержка: {(torch_time/1000)*1000:.2f} ms/sample")
    
    # 2. ONNX модель
    print("\n2. ONNX модель (CPU):")
    ort_session = ort.InferenceSession(config['model_paths']['onnx'])
    
    start = time.time()
    for i in range(0, 1000, 100):
        batch = dummy_input_np[i:i+100]
        _ = ort_session.run(None, {'input': batch})
    onnx_time = time.time() - start
    
    results.append({
        'model': 'ONNX',
        'samples_per_sec': 1000 / onnx_time,
        'latency_ms': (onnx_time / 1000) * 1000,
        'speedup_vs_pytorch': torch_time / onnx_time
    })
    print(f"   Скорость: {1000/onnx_time:.2f} samples/sec")
    print(f"   Задержка: {(onnx_time/1000)*1000:.2f} ms/sample")
    print(f"   Ускорение: {torch_time/onnx_time:.2f}x")
    
    # Сохранение результатов
    benchmark_path = 'reports/benchmark_results.json'
    Path('reports').mkdir(exist_ok=True)
    
    with open(benchmark_path, 'w') as f:
        json.dump({
            'benchmark_results': results,
            'timestamp': datetime.now().isoformat(),
            'hardware': {
                'cpu_cores': torch.get_num_threads(),
                'device': 'CPU'
            },
            'summary': {
                'best_performance': max(results, key=lambda x: x['samples_per_sec'])['model'],
                'pytorch_to_onnx_speedup': torch_time / onnx_time if 'onnx_time' in locals() else None
            }
        }, f, indent=2)
    
    print(f"\n📊 Отчет сохранен: {benchmark_path}")
    
    return results

# ==================== 6. ОПТИМИЗАЦИЯ ====================
def optimize_model(config, input_size):
    """Оптимизация модели (Quantization/Pruning)"""
    print("\n" + "=" * 60)
    print("📉 ОПТИМИЗАЦИЯ МОДЕЛИ")
    print("=" * 60)
    
    # Загрузка модели
    model = CreditScoringNN(input_size)
    model.load_state_dict(torch.load(config['model_paths']['final_model']))
    model.eval()
    
    # 1. Dynamic Quantization (8-bit)
    print("\n1. Dynamic Quantization (INT8):")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    # Сохранение квантованной модели
    quantized_path = config['model_paths']['final_model'].replace('.pth', '_quantized.pth')
    torch.jit.save(torch.jit.script(quantized_model), quantized_path)
    
    # Сравнение размеров
    original_size = Path(config['model_paths']['final_model']).stat().st_size / 1024
    quantized_size = Path(quantized_path).stat().st_size / 1024
    
    print(f"   Оригинальный размер: {original_size:.1f} KB")
    print(f"   Квантованный размер: {quantized_size:.1f} KB")
    print(f"   Сжатие: {original_size/quantized_size:.1f}x")
    
    # 2. Pruning (Удаление весов)
    print("\n2. Pruning (Удаление 20% весов):")
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Применяем pruning
    from torch.nn.utils import prune
    for module, param_name in parameters_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=0.2)
    
    # Сохранение pruned модели
    pruned_path = config['model_paths']['final_model'].replace('.pth', '_pruned.pth')
    torch.save(model.state_dict(), pruned_path)
    
    # Тестирование pruned модели
    model.eval()
    dummy_input = torch.randn(1, input_size)
    with torch.no_grad():
        start = time.time()
        for _ in range(1000):
            _ = model(dummy_input)
        pruned_time = time.time() - start
    
    print(f"   Время инференса (pruned): {pruned_time/1000*1000:.2f} ms/sample")
    
    # Сохранение отчета об оптимизации
    optimization_report = {
        'quantization': {
            'original_size_kb': original_size,
            'quantized_size_kb': quantized_size,
            'compression_ratio': original_size / quantized_size
        },
        'pruning': {
            'pruned_parameters_percentage': 20,
            'inference_time_ms': pruned_time / 1000 * 1000
        },
        'timestamp': datetime.now().isoformat()
    }
    
    optimization_path = 'reports/optimization_report.json'
    with open(optimization_path, 'w') as f:
        json.dump(optimization_report, f, indent=2)
    
    print(f"\n📊 Отчет оптимизации: {optimization_path}")
    
    return quantized_path, pruned_path

# ==================== 7. ГЛАВНЫЙ ПАЙПЛАЙН ====================
def main():
    """Главный пайплайн"""
    print("=" * 60)
    print("🎯 ПРОМЫШЛЕННЫЙ ПАЙПЛАЙН ОБУЧЕНИЯ")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/training_config.yaml')
    parser.add_argument('--skip-optimization', action='store_true')
    args = parser.parse_args()
    
    # Загрузка конфигурации
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Создание директорий
    Path('models/trained').mkdir(parents=True, exist_ok=True)
    Path('models/processed').mkdir(parents=True, exist_ok=True)
    Path('reports').mkdir(parents=True, exist_ok=True)
    
    # 1. Обучение нейронной сети
    model, input_size = train_neural_network(config)
    
    # 2. Конвертация в ONNX
    onnx_path = convert_to_onnx(model, input_size, config)
    
    # 3. Бенчмаркинг
    benchmark_results = benchmark_models(config, input_size)
    
    # 4. Оптимизация (если не пропущена)
    if not args.skip_optimization:
        quantized_path, pruned_path = optimize_model(config, input_size)
    
    # 5. Генерация финального отчета
    print("\n" + "=" * 60)
    print("📋 ФИНАЛЬНЫЙ ОТЧЕТ")
    print("=" * 60)
    
    # Загрузка метрик
    with open(config['model_paths']['metrics'], 'r') as f:
        metrics = json.load(f)
    
    print(f"\n📈 МЕТРИКИ МОДЕЛИ:")
    print(f"   ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"   Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   F1-Score: {metrics['test_f1_score']:.4f}")
    
    print(f"\n💾 ФАЙЛЫ МОДЕЛИ:")
    print(f"   PyTorch: {config['model_paths']['final_model']}")
    print(f"   ONNX: {onnx_path}")
    print(f"   Best model: {config['model_paths']['best_model']}")
    print(f"   Метрики: {config['model_paths']['metrics']}")
    
    print(f"\n📊 ОТЧЕТЫ:")
    print(f"   Бенчмарк: reports/benchmark_results.json")
    if not args.skip_optimization:
        print(f"   Оптимизация: reports/optimization_report.json")
    
    print("\n✅ ПАЙПЛАЙН ВЫПОЛНЕН УСПЕШНО!")
    print("=" * 60)

if __name__ == "__main__":
    main()