"""
Оптимизация моделей: Quantization и Pruning
Этап 1: Подготовка модели к промышленной эксплуатации
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import onnx
import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
import time

class ModelOptimizer:
    def __init__(self, model_path, input_size):
        self.model_path = model_path
        self.input_size = input_size
        
    def dynamic_quantization(self, model):
        """Динамическое квантование модели в 8-bit"""
        print("🔧 Применение динамического квантования (INT8)...")
        
        # Динамическое квантование
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.BatchNorm1d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def static_quantization(self, model, calibration_data):
        """Статическое квантование с калибровкой"""
        print("🔧 Применение статического квантования (INT8)...")
        
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Подготовка модели
        model_prepared = torch.quantization.prepare(model)
        
        # Калибровка
        print("   Калибровка на 100 примерах...")
        with torch.no_grad():
            for i in range(100):
                dummy_input = calibration_data[i:i+1]
                _ = model_prepared(dummy_input)
        
        # Конвертация
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized
    
    def apply_pruning(self, model, pruning_rate=0.2):
        """Применение pruning для уменьшения модели"""
        print(f"✂️  Применение pruning ({pruning_rate*100}% весов)...")
        
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        # Применяем L1 unstructured pruning
        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=pruning_rate)
        
        # Удаляем маски pruning для постоянного эффекта
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def optimize_onnx_model(self, onnx_path):
        """Оптимизация ONNX модели"""
        print("🔧 Оптимизация ONNX модели...")
        
        # Загрузка ONNX модели
        model = onnx.load(onnx_path)
        
        # Базовые оптимизации
        from onnxruntime.transformers import optimizer
        
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type='bert',
            num_heads=1,
            hidden_size=self.input_size
        )
        
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        optimized_model.save_model_to_file(optimized_path)
        
        # Сравнение размеров
        original_size = Path(onnx_path).stat().st_size / 1024
        optimized_size = Path(optimized_path).stat().st_size / 1024
        
        print(f"   Оригинальный размер: {original_size:.1f} KB")
        print(f"   Оптимизированный размер: {optimized_size:.1f} KB")
        print(f"   Уменьшение: {(1 - optimized_size/original_size)*100:.1f}%")
        
        return optimized_path
    
    def benchmark_optimization(self, original_model, optimized_model, test_data):
        """Бенчмаркинг оптимизированной модели"""
        print("📊 Бенчмаркинг оптимизированной модели...")
        
        results = []
        
        # Оригинальная модель
        original_model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(0, len(test_data), 100):
                batch = test_data[i:i+100]
                _ = original_model(batch)
        original_time = time.time() - start
        
        # Оптимизированная модель
        optimized_model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(0, len(test_data), 100):
                batch = test_data[i:i+100]
                _ = optimized_model(batch)
        optimized_time = time.time() - start
        
        # Сравнение точности (если есть метки)
        if hasattr(test_data, 'labels'):
            with torch.no_grad():
                outputs_original = original_model(test_data.features[:100])
                outputs_optimized = optimized_model(test_data.features[:100])
            
            mae = torch.mean(torch.abs(outputs_original - outputs_optimized)).item()
            print(f"   Средняя ошибка предсказаний: {mae:.6f}")
        else:
            mae = None
        
        results = {
            'original_inference_time_ms': original_time / len(test_data) * 1000,
            'optimized_inference_time_ms': optimized_time / len(test_data) * 1000,
            'speedup_ratio': original_time / optimized_time,
            'mean_absolute_error': mae,
            'memory_reduction_percentage': None  # Нужно сравнить размеры файлов
        }
        
        print(f"   Ускорение инференса: {results['speedup_ratio']:.2f}x")
        
        return results
    
    def create_optimization_report(self, results, output_path='reports/optimization_report.json'):
        """Создание отчета об оптимизации"""
        report = {
            'optimization_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_size': self.input_size,
            'optimization_results': results,
            'recommendations': []
        }
        
        if results.get('speedup_ratio', 1) > 1.5:
            report['recommendations'].append("✅ Значительное ускорение инференса достигнуто")
        
        if results.get('mean_absolute_error', 0) < 0.01:
            report['recommendations'].append("✅ Точность сохранилась после оптимизации")
        
        # Сохранение отчета
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Отчет сохранен: {output_path}")
        
        return report

def main():
    """Пример использования"""
    import yaml
    from src.ml_pipeline.training.train_model import CreditScoringNN
    
    # Загрузка конфигурации
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Загрузка модели
    input_size = 20  # Пример: количество признаков
    model = CreditScoringNN(input_size)
    model.load_state_dict(torch.load(config['model_paths']['final_model']))
    
    # Оптимизатор
    optimizer = ModelOptimizer(
        model_path=config['model_paths']['final_model'],
        input_size=input_size
    )
    
    # 1. Pruning
    pruned_model = optimizer.apply_pruning(model, pruning_rate=0.2)
    
    # 2. Dynamic Quantization
    quantized_model = optimizer.dynamic_quantization(pruned_model)
    
    # 3. Бенчмаркинг
    test_data = torch.randn(1000, input_size)
    results = optimizer.benchmark_optimization(model, quantized_model, test_data)
    
    # 4. Отчет
    optimizer.create_optimization_report(results)
    
    print("✅ Оптимизация завершена!")

if __name__ == "__main__":
    main()