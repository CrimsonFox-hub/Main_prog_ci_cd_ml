"""
Конвертация PyTorch модели в ONNX формат
"""
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any

from src.utils.logger import model_logger
from src.ml_pipeline.training.train_model import CreditScoringNN

class ModelConverter:
    """Конвертер моделей в ONNX формат"""
    
    def __init__(self, model_path: str, input_size: int):
        self.model_path = Path(model_path)
        self.input_size = input_size
        
    def convert_to_onnx(self, output_path: str = None, opset_version: int = 13) -> str:
        """Конвертация модели в ONNX"""
        if output_path is None:
            output_path = self.model_path.with_suffix('.onnx')
        
        # Загрузка модели PyTorch
        model = CreditScoringNN(input_size=self.input_size)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model.eval()
        
        # Создание примера входных данных
        dummy_input = torch.randn(1, self.input_size)
        
        # Экспорт в ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Проверка корректности экспорта
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        model_logger.info(f"Model converted to ONNX: {output_path}")
        model_logger.info(f"Input shape: {dummy_input.shape}")
        
        return str(output_path)
    
    def validate_conversion(self, onnx_path: str, num_samples: int = 100) -> Dict[str, Any]:
        """Валидация корректности конвертации"""
        model_logger.info("Validating ONNX conversion...")
        
        # Загрузка оригинальной модели PyTorch
        pt_model = CreditScoringNN(input_size=self.input_size)
        pt_model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        pt_model.eval()
        
        # Загрузка ONNX модели
        ort_session = ort.InferenceSession(onnx_path)
        
        # Генерация тестовых данных
        test_inputs = torch.randn(num_samples, self.input_size)
        
        # Предсказания PyTorch
        pt_predictions = []
        with torch.no_grad():
            for i in range(num_samples):
                pt_output = pt_model(test_inputs[i:i+1])
                pt_predictions.append(pt_output.numpy())
        
        # Предсказания ONNX Runtime
        onnx_predictions = []
        for i in range(num_samples):
            ort_inputs = {ort_session.get_inputs()[0].name: test_inputs[i:i+1].numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]
            onnx_predictions.append(ort_output)
        
        # Сравнение результатов
        pt_array = np.concatenate(pt_predictions, axis=0)
        onnx_array = np.concatenate(onnx_predictions, axis=0)
        
        # Расчет метрик сравнения
        abs_diff = np.abs(pt_array - onnx_array)
        mae = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
        mse = np.mean((pt_array - onnx_array) ** 2)
        
        validation_results = {
            'mae': float(mae),
            'max_diff': float(max_diff),
            'mse': float(mse),
            'num_samples': num_samples,
            'conversion_valid': mae < 1e-4,  # Порог допустимой ошибки
            'pt_shape': pt_array.shape,
            'onnx_shape': onnx_array.shape
        }
        
        model_logger.info(f"Validation results: {validation_results}")
        
        if validation_results['conversion_valid']:
            model_logger.info("✅ ONNX conversion validated successfully!")
        else:
            model_logger.warning("⚠️ ONNX conversion validation failed!")
        
        return validation_results
    
    def benchmark_performance(self, onnx_path: str, pt_model_path: str = None) -> Dict[str, Any]:
        """Сравнение производительности PyTorch и ONNX Runtime"""
        model_logger.info("Benchmarking performance...")
        
        if pt_model_path is None:
            pt_model_path = self.model_path
        
        # Загрузка моделей
        pt_model = CreditScoringNN(input_size=self.input_size)
        pt_model.load_state_dict(torch.load(pt_model_path, map_location='cpu'))
        pt_model.eval()
        
        ort_session = ort.InferenceSession(onnx_path)
        
        # Тестовые данные
        batch_sizes = [1, 8, 16, 32, 64, 128]
        num_iterations = 100
        
        benchmark_results = {
            'pytorch': {},
            'onnx': {}
        }
        
        for batch_size in batch_sizes:
            model_logger.info(f"Benchmarking batch size: {batch_size}")
            
            # Генерация данных
            test_input = torch.randn(batch_size, self.input_size)
            
            # PyTorch benchmark
            pt_times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    _ = pt_model(test_input)
                    end_time = time.perf_counter()
                    pt_times.append((end_time - start_time) * 1000)  # мс
            
            # ONNX Runtime benchmark
            onnx_times = []
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = ort_session.run(None, ort_inputs)
                end_time = time.perf_counter()
                onnx_times.append((end_time - start_time) * 1000)  # мс
            
            # Расчет статистики
            benchmark_results['pytorch'][batch_size] = {
                'mean_ms': np.mean(pt_times),
                'std_ms': np.std(pt_times),
                'p95_ms': np.percentile(pt_times, 95),
                'throughput_rps': batch_size / (np.mean(pt_times) / 1000)
            }
            
            benchmark_results['onnx'][batch_size] = {
                'mean_ms': np.mean(onnx_times),
                'std_ms': np.std(onnx_times),
                'p95_ms': np.percentile(onnx_times, 95),
                'throughput_rps': batch_size / (np.mean(onnx_times) / 1000)
            }
        
        # Сравнение производительности
        comparison = {}
        for batch_size in batch_sizes:
            pt_mean = benchmark_results['pytorch'][batch_size]['mean_ms']
            onnx_mean = benchmark_results['onnx'][batch_size]['mean_ms']
            
            comparison[batch_size] = {
                'speedup': pt_mean / max(onnx_mean, 1e-6),
                'pt_throughput': benchmark_results['pytorch'][batch_size]['throughput_rps'],
                'onnx_throughput': benchmark_results['onnx'][batch_size]['throughput_rps']
            }
        
        # Сохранение результатов
        results = {
            'benchmark_results': benchmark_results,
            'comparison': comparison,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware': {
                'cpu': torch.get_num_threads(),
                'device': 'CPU'
            }
        }
        
        # Сохранение в файл
        results_path = Path('reports/onnx_benchmark.json')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        model_logger.info(f"Benchmark results saved to: {results_path}")
        
        return results

def main():
    """Пример использования конвертера"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, default='models/credit_scoring.pth',
                       help='Path to PyTorch model')
    parser.add_argument('--input_size', type=int, default=20,
                       help='Input size for the model')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for ONNX model')
    
    args = parser.parse_args()
    
    # Конвертация
    converter = ModelConverter(args.model_path, args.input_size)
    onnx_path = converter.convert_to_onnx(args.output_path)
    
    # Валидация
    validation_results = converter.validate_conversion(onnx_path)
    
    # Бенчмаркинг
    benchmark_results = converter.benchmark_performance(onnx_path)
    
    print(f"\n{'='*50}")
    print("CONVERSION SUMMARY:")
    print(f"{'='*50}")
    print(f"✅ ONNX model saved: {onnx_path}")
    print(f"✅ Validation MAE: {validation_results['mae']:.6f}")
    print(f"✅ Conversion valid: {validation_results['conversion_valid']}")
    
    # Показать сравнение производительности для batch_size=32
    if 32 in benchmark_results['comparison']:
        comp = benchmark_results['comparison'][32]
        print(f"📊 Performance comparison (batch_size=32):")
        print(f"   PyTorch throughput: {comp['pt_throughput']:.1f} RPS")
        print(f"   ONNX throughput: {comp['onnx_throughput']:.1f} RPS")
        print(f"   Speedup: {comp['speedup']:.2f}x")

if __name__ == "__main__":
    main()