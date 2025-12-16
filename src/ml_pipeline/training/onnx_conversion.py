"""
Конвертация модели в ONNX и оптимизация
Этап 1: Подготовка модели к промышленной эксплуатации
"""
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Tuple
import yaml

class ModelConverter:
    """Класс для конвертации и оптимизации моделей"""
    
    def __init__(self, config_path: str = 'configs/training_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pytorch_model(self) -> torch.nn.Module:
        """Загрузка PyTorch модели"""
        from train_model import CreditScoringNN
        
        # Загрузка скейлера для определения размера входа
        import joblib
        scaler = joblib.load(self.config['model_paths']['scaler'])
        input_size = scaler.n_features_in_
        
        # Создание и загрузка модели
        model = CreditScoringNN(
            input_size=input_size,
            hidden_layers=self.config['model']['hidden_layers']
        )
        
        model.load_state_dict(
            torch.load(self.config['model_paths']['best_model'], map_location=self.device)
        )
        model.eval()
        
        return model, input_size
    
    def convert_to_onnx(self, model: torch.nn.Module, input_size: int) -> str:
        """Конвертация PyTorch модели в ONNX"""
        onnx_path = self.config['model_paths']['onnx_model']
        
        # Создание примера входных данных
        dummy_input = torch.randn(1, input_size, device=self.device)
        
        # Экспорт в ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        # Валидация ONNX модели
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"Model successfully converted to ONNX: {onnx_path}")
        return onnx_path
    
    def quantize_onnx_model(self, onnx_path: str) -> str:
        """Квантование ONNX модели для уменьшения размера"""
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = onnx_path.replace('.onnx', '_quantized.onnx')
        
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        
        # Проверка размера моделей
        original_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
        quantized_size = Path(quantized_path).stat().st_size / (1024 * 1024)  # MB
        
        print(f"Original ONNX size: {original_size:.2f} MB")
        print(f"Quantized ONNX size: {quantized_size:.2f} MB")
        print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
        
        return quantized_path
    
    def prune_model(self, model: torch.nn.Module, amount: float = 0.2) -> torch.nn.Module:
        """Прунинг модели для ускорения инференса"""
        import torch.nn.utils.prune as prune
        
        # Применение прунинга к линейным слоям
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')  # Удаление маски, веса становятся разреженными
        
        return model
    
    def benchmark_inference(self, model, onnx_path: str) -> Dict:
        """Бенчмаркинг производительности инференса"""
        results = {}
        
        # Подготовка тестовых данных
        test_data = torch.randn(1000, model.hidden_layers[0].in_features).to(self.device)
        
        # Бенчмарк PyTorch
        print("Benchmarking PyTorch inference...")
        
        # Warmup
        for _ in range(10):
            _ = model(test_data[:10])
        
        # Измерение времени
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, len(test_data), 32):
                batch = test_data[i:i+32]
                _ = model(batch)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        pytorch_time = time.time() - start_time
        
        # Бенчмарк ONNX
        print("Benchmarking ONNX inference...")
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider' if self.device.type == 'cuda' else 'CPUExecutionProvider']
        )
        
        input_name = ort_session.get_inputs()[0].name
        test_data_np = test_data.cpu().numpy().astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = ort_session.run(None, {input_name: test_data_np[:10]})
        
        # Измерение времени
        start_time = time.time()
        
        for i in range(0, len(test_data_np), 32):
            batch = test_data_np[i:i+32]
            _ = ort_session.run(None, {input_name: batch})
        
        onnx_time = time.time() - start_time
        
        # Сравнение производительности
        speedup = pytorch_time / onnx_time
        
        results = {
            'pytorch_inference_time': pytorch_time,
            'onnx_inference_time': onnx_time,
            'speedup_factor': speedup,
            'samples_per_second_pytorch': len(test_data) / pytorch_time,
            'samples_per_second_onnx': len(test_data) / onnx_time
        }
        
        print(f"\n=== Benchmark Results ===")
        print(f"PyTorch: {pytorch_time:.3f}s ({len(test_data)/pytorch_time:.1f} samples/s)")
        print(f"ONNX: {onnx_time:.3f}s ({len(test_data)/onnx_time:.1f} samples/s)")
        print(f"Speedup: {speedup:.2f}x")
        
        return results
    
    def validate_conversion(self, model: torch.nn.Module, onnx_path: str) -> bool:
        """Валидация корректности конвертации"""
        print("Validating model conversion...")
        
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        
        # Тестирование на случайных данных
        test_input = torch.randn(10, model.hidden_layers[0].in_features)
        
        # PyTorch вывод
        model.eval()
        with torch.no_grad():
            pytorch_output = model(test_input).numpy()
        
        # ONNX вывод
        onnx_output = ort_session.run(
            None, 
            {input_name: test_input.numpy().astype(np.float32)}
        )[0]
        
        # Сравнение результатов
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-4 and mean_diff < 1e-5:
            print("✓ Conversion validation PASSED")
            return True
        else:
            print("✗ Conversion validation FAILED")
            return False
    
    def run_benchmarks_on_resources(self) -> Dict:
        """Нагрузочное тестирование на разных типах инстансов"""
        print("\n=== Resource Benchmarking ===")
        
        benchmarks = {}
        test_data = np.random.randn(10000, 20).astype(np.float32)
        
        # Тестирование на CPU
        cpu_session = ort.InferenceSession(
            self.config['model_paths']['onnx_model'],
            providers=['CPUExecutionProvider']
        )
        
        start_time = time.time()
        for _ in range(100):
            _ = cpu_session.run(None, {'input': test_data[:100]})
        cpu_time = time.time() - start_time
        
        benchmarks['cpu'] = {
            'inference_time': cpu_time,
            'throughput': 10000 / cpu_time
        }
        
        # Тестирование на GPU (если доступно)
        if torch.cuda.is_available():
            gpu_session = ort.InferenceSession(
                self.config['model_paths']['onnx_model'],
                providers=['CUDAExecutionProvider']
            )
            
            start_time = time.time()
            for _ in range(100):
                _ = gpu_session.run(None, {'input': test_data[:100]})
            gpu_time = time.time() - start_time
            
            benchmarks['gpu'] = {
                'inference_time': gpu_time,
                'throughput': 10000 / gpu_time,
                'speedup_vs_cpu': cpu_time / gpu_time
            }
        
        # Генерация отчета
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmarks': benchmarks,
            'recommended_config': self._get_recommended_config(benchmarks)
        }
        
        # Сохранение отчета
        report_path = 'reports/benchmark_report.json'
        Path('reports').mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Benchmark report saved to: {report_path}")
        return report
    
    def _get_recommended_config(self, benchmarks: Dict) -> Dict:
        """Определение оптимальной конфигурации ресурсов"""
        recommendations = {
            'development': {
                'instance_type': 'CPU',
                'reason': 'Cost-effective for development and testing'
            },
            'staging': {
                'instance_type': 'CPU with auto-scaling',
                'min_replicas': 2,
                'max_replicas': 5,
                'reason': 'Balanced performance and cost for staging'
            }
        }
        
        if 'gpu' in benchmarks:
            gpu_speedup = benchmarks['gpu'].get('speedup_vs_cpu', 1)
            
            if gpu_speedup > 5:  # GPU значительно быстрее
                recommendations['production'] = {
                    'instance_type': 'GPU (V100 or better)',
                    'min_replicas': 2,
                    'max_replicas': 10,
                    'autoscaling_metric': 'concurrent_requests',
                    'target_utilization': 70,
                    'reason': f'GPU provides {gpu_speedup:.1f}x speedup over CPU'
                }
            else:
                recommendations['production'] = {
                    'instance_type': 'High-CPU instance with auto-scaling',
                    'min_replicas': 3,
                    'max_replicas': 15,
                    'reason': 'CPU provides sufficient performance at lower cost'
                }
        
        return recommendations
    
    def run_full_pipeline(self):
        """Запуск полного пайплайна конвертации и оптимизации"""
        print("Starting model conversion and optimization pipeline...")
        
        # 1. Загрузка модели
        model, input_size = self.load_pytorch_model()
        
        # 2. Конвертация в ONNX
        onnx_path = self.convert_to_onnx(model, input_size)
        
        # 3. Валидация конвертации
        is_valid = self.validate_conversion(model, onnx_path)
        
        if not is_valid:
            print("Warning: Model conversion validation failed!")
        
        # 4. Квантование
        quantized_path = self.quantize_onnx_model(onnx_path)
        
        # 5. Прунинг (опционально)
        if self.config['optimization'].get('enable_pruning', False):
            print("Applying model pruning...")
            model = self.prune_model(model, amount=0.2)
            pruned_onnx_path = onnx_path.replace('.onnx', '_pruned.onnx')
            self.convert_to_onnx(model, input_size, pruned_onnx_path)
        
        # 6. Бенчмаркинг
        benchmark_results = self.benchmark_inference(model, onnx_path)
        
        # 7. Нагрузочное тестирование
        resource_benchmarks = self.run_benchmarks_on_resources()
        
        # 8. Генерация итогового отчета
        final_report = {
            'conversion': {
                'status': 'SUCCESS' if is_valid else 'WARNING',
                'onnx_model_path': onnx_path,
                'quantized_model_path': quantized_path,
                'validation_passed': is_valid
            },
            'optimization': {
                'original_size_mb': Path(onnx_path).stat().st_size / (1024 * 1024),
                'quantized_size_mb': Path(quantized_path).stat().st_size / (1024 * 1024),
                'size_reduction_percent': None
            },
            'performance': benchmark_results,
            'resource_benchmarks': resource_benchmarks
        }
        
        # Расчет сокращения размера
        original_size = final_report['optimization']['original_size_mb']
        quantized_size = final_report['optimization']['quantized_size_mb']
        final_report['optimization']['size_reduction_percent'] = (
            (original_size - quantized_size) / original_size * 100
        )
        
        # Сохранение отчета
        with open('reports/conversion_final_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print("\n" + "="*50)
        print("CONVERSION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        
        return final_report

if __name__ == "__main__":
    converter = ModelConverter()
    report = converter.run_full_pipeline()