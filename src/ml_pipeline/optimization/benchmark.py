"""
Бенчмаркинг оптимизированных моделей
"""
import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import psutil
import GPUtil

from src.utils.logger import model_logger
from src.ml_pipeline.inference.predictor import ModelPredictor
from src.ml_pipeline.optimization.model_optimizer import ModelOptimizer

class ModelBenchmark:
    """Класс для бенчмаркинга моделей"""
    
    def __init__(self, model_paths: Dict[str, str]):
        self.model_paths = model_paths
        self.results = {}
        
        # Конфигурация тестов
        self.batch_sizes = [1, 8, 16, 32, 64, 128]
        self.test_duration = 10  # секунд на каждый тест
        self.warmup_iterations = 100
    
    def benchmark_models(self) -> Dict[str, Any]:
        """Запуск полного бенчмарка для всех моделей"""
        model_logger.info("Starting model benchmarking")
        
        for model_name, model_path in self.model_paths.items():
            model_logger.info(f"Benchmarking model: {model_name}")
            
            try:
                # Загрузка модели
                predictor = ModelPredictor(model_path)
                
                # Бенчмарк для разных batch sizes
                model_results = {}
                
                for batch_size in self.batch_sizes:
                    batch_result = self._benchmark_batch_size(
                        predictor, batch_size
                    )
                    model_results[f"batch_{batch_size}"] = batch_result
                
                # Сводные метрики
                model_results['summary'] = self._create_summary(model_results)
                
                self.results[model_name] = model_results
                
                model_logger.info(f"Benchmark completed for {model_name}")
                
            except Exception as e:
                model_logger.error(f"Benchmark failed for {model_name}: {e}", exc_info=True)
                self.results[model_name] = {'error': str(e)}
        
        # Сравнение моделей
        comparison = self._compare_models()
        self.results['comparison'] = comparison
        
        # Сохранение результатов
        self.save_results()
        
        return self.results
    
    def _benchmark_batch_size(self, predictor: ModelPredictor, 
                             batch_size: int) -> Dict[str, Any]:
        """Бенчмарк для конкретного размера батча"""
        model_logger.debug(f"Benchmarking batch size: {batch_size}")
        
        # Генерация тестовых данных
        test_features = self._generate_test_features(batch_size * 100)
        
        # Warmup
        for i in range(self.warmup_iterations):
            batch = test_features[i * batch_size:(i + 1) * batch_size]
            _ = predictor.batch_predict(batch)
        
        # Измерение производительности
        start_time = time.time()
        iterations = 0
        total_predictions = 0
        
        memory_before = psutil.Process().memory_info().rss
        
        while time.time() - start_time < self.test_duration:
            batch = test_features[iterations * batch_size:(iterations + 1) * batch_size]
            
            batch_start = time.time()
            predictions = predictor.batch_predict(batch)
            batch_time = time.time() - batch_start
            
            iterations += 1
            total_predictions += len(batch)
        
        total_time = time.time() - start_time
        memory_after = psutil.Process().memory_info().rss
        
        # Расчет метрик
        throughput = total_predictions / total_time
        avg_latency = total_time / total_predictions * 1000  # мс
        
        # Процентили латентности (если есть индивидуальные измерения)
        latency_percentiles = self._calculate_latency_percentiles(
            predictor, test_features[:1000], batch_size
        )
        
        return {
            'batch_size': batch_size,
            'throughput_rps': throughput,
            'avg_latency_ms': avg_latency,
            'total_predictions': total_predictions,
            'total_time_seconds': total_time,
            'memory_usage_mb': (memory_after - memory_before) / 1024 / 1024,
            'latency_percentiles_ms': latency_percentiles,
            'iterations': iterations
        }
    
    def _generate_test_features(self, n_samples: int) -> List[Dict[str, Any]]:
        """Генерация тестовых признаков"""
        features_list = []
        
        for i in range(n_samples):
            features = {
                'age': np.random.randint(18, 80),
                'income': np.random.uniform(20000, 200000),
                'credit_score': np.random.randint(300, 850),
                'loan_amount': np.random.uniform(1000, 50000),
                'employment_years': np.random.randint(0, 40),
                'debt_to_income': np.random.uniform(0, 1),
                'has_default': np.random.choice([True, False]),
                'loan_purpose': np.random.choice(['car', 'home', 'education', 'business'])
            }
            features_list.append(features)
        
        return features_list
    
    def _calculate_latency_percentiles(self, predictor: ModelPredictor,
                                      test_features: List[Dict[str, Any]],
                                      batch_size: int) -> Dict[str, float]:
        """Расчет процентилей латентности"""
        latencies = []
        
        # Измерение латентности для отдельных запросов
        for i in range(0, min(1000, len(test_features)), batch_size):
            batch = test_features[i:i + batch_size]
            
            start_time = time.perf_counter()
            _ = predictor.batch_predict(batch)
            end_time = time.perf_counter()
            
            batch_latency = (end_time - start_time) * 1000 / len(batch)  # мс на запрос
            latencies.append(batch_latency)
        
        if not latencies:
            return {}
        
        percentiles = [50, 75, 90, 95, 99]
        results = {}
        
        for p in percentiles:
            results[f'p{p}'] = float(np.percentile(latencies, p))
        
        return results
    
    def _create_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Создание сводки по результатам модели"""
        best_throughput = 0
        best_latency = float('inf')
        optimal_batch_size = 1
        
        for batch_key, result in model_results.items():
            if batch_key.startswith('batch_'):
                throughput = result.get('throughput_rps', 0)
                latency = result.get('avg_latency_ms', float('inf'))
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    optimal_batch_size = result['batch_size']
                
                if latency < best_latency:
                    best_latency = latency
        
        return {
            'best_throughput_rps': best_throughput,
            'best_latency_ms': best_latency,
            'optimal_batch_size': optimal_batch_size,
            'throughput_at_optimal': model_results.get(
                f'batch_{optimal_batch_size}', {}
            ).get('throughput_rps', 0)
        }
    
    def _compare_models(self) -> Dict[str, Any]:
        """Сравнение производительности разных моделей"""
        if len(self.results) < 2:
            return {}
        
        comparison = {}
        baseline_model = list(self.results.keys())[0]
        
        for model_name, results in self.results.items():
            if model_name == 'comparison' or 'error' in results:
                continue
            
            baseline_summary = self.results[baseline_model].get('summary', {})
            model_summary = results.get('summary', {})
            
            if baseline_summary and model_summary:
                throughput_improvement = (
                    model_summary.get('best_throughput_rps', 0) / 
                    max(baseline_summary.get('best_throughput_rps', 1), 1)
                )
                
                latency_improvement = (
                    baseline_summary.get('best_latency_ms', 1) / 
                    max(model_summary.get('best_latency_ms', 1), 1)
                )
                
                comparison[model_name] = {
                    'throughput_vs_baseline': throughput_improvement,
                    'latency_vs_baseline': latency_improvement,
                    'optimal_batch_size': model_summary.get('optimal_batch_size'),
                    'recommendation': self._generate_recommendation(
                        throughput_improvement, latency_improvement
                    )
                }
        
        return comparison
    
    def _generate_recommendation(self, throughput_improvement: float,
                                latency_improvement: float) -> str:
        """Генерация рекомендации на основе результатов"""
        if throughput_improvement > 1.5 and latency_improvement > 1.5:
            return "Highly recommended - significant improvement in both throughput and latency"
        elif throughput_improvement > 1.2:
            return "Recommended for high-throughput scenarios"
        elif latency_improvement > 1.2:
            return "Recommended for low-latency scenarios"
        elif throughput_improvement > 1.0 and latency_improvement > 1.0:
            return "Minor improvements, consider based on specific requirements"
        else:
            return "Not recommended - no significant improvements"
    
    def benchmark_hardware(self) -> Dict[str, Any]:
        """Бенчмарк аппаратного обеспечения"""
        hardware_info = {}
        
        try:
            # CPU информация
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'usage_percent': psutil.cpu_percent(interval=1)
            }
            hardware_info['cpu'] = cpu_info
            
            # Память
            memory = psutil.virtual_memory()
            memory_info = {
                'total_gb': memory.total / 1024**3,
                'available_gb': memory.available / 1024**3,
                'percent': memory.percent
            }
            hardware_info['memory'] = memory_info
            
            # GPU (если есть)
            try:
                gpus = GPUtil.getGPUs()
                gpu_info = []
                
                for gpu in gpus:
                    gpu_info.append({
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_free_mb': gpu.memoryFree,
                        'load_percent': gpu.load * 100,
                        'temperature_c': gpu.temperature
                    })
                
                if gpu_info:
                    hardware_info['gpu'] = gpu_info
                    
            except:
                hardware_info['gpu'] = 'No GPU detected'
            
            # Диск
            disk = psutil.disk_usage('/')
            disk_info = {
                'total_gb': disk.total / 1024**3,
                'used_gb': disk.used / 1024**3,
                'free_gb': disk.free / 1024**3,
                'percent': disk.percent
            }
            hardware_info['disk'] = disk_info
            
            model_logger.info("Hardware benchmarking completed")
            
        except Exception as e:
            model_logger.error(f"Hardware benchmark failed: {e}", exc_info=True)
            hardware_info['error'] = str(e)
        
        return hardware_info
    
    def run_stress_test(self, model_name: str, duration: int = 300) -> Dict[str, Any]:
        """Стресс-тест модели"""
        model_logger.info(f"Starting stress test for {model_name} (duration: {duration}s)")
        
        if model_name not in self.model_paths:
            return {'error': f'Model {model_name} not found'}
        
        try:
            predictor = ModelPredictor(self.model_paths[model_name])
            
            # Генерация тестовых данных
            test_features = self._generate_test_features(10000)
            
            start_time = time.time()
            total_predictions = 0
            errors = 0
            
            # Мониторинг ресурсов
            cpu_usages = []
            memory_usages = []
            
            while time.time() - start_time < duration:
                # Измерение использования ресурсов
                cpu_usages.append(psutil.cpu_percent(interval=0.1))
                memory_usages.append(psutil.Process().memory_info().rss / 1024**2)
                
                # Выполнение предсказаний
                batch_size = 32
                batch = test_features[:batch_size]
                
                try:
                    _ = predictor.batch_predict(batch)
                    total_predictions += len(batch)
                except Exception as e:
                    errors += 1
                    model_logger.error(f"Stress test error: {e}")
            
            total_time = time.time() - start_time
            
            # Анализ использования ресурсов
            avg_cpu = np.mean(cpu_usages) if cpu_usages else 0
            max_cpu = np.max(cpu_usages) if cpu_usages else 0
            avg_memory = np.mean(memory_usages) if memory_usages else 0
            max_memory = np.max(memory_usages) if memory_usages else 0
            
            result = {
                'duration_seconds': total_time,
                'total_predictions': total_predictions,
                'throughput_rps': total_predictions / total_time,
                'errors': errors,
                'error_rate': errors / max(total_predictions, 1),
                'resource_usage': {
                    'avg_cpu_percent': avg_cpu,
                    'max_cpu_percent': max_cpu,
                    'avg_memory_mb': avg_memory,
                    'max_memory_mb': max_memory
                },
                'stability': 'stable' if errors == 0 else 'unstable'
            }
            
            model_logger.info(f"Stress test completed: {result['throughput_rps']:.1f} RPS")
            
            return result
            
        except Exception as e:
            model_logger.error(f"Stress test failed: {e}", exc_info=True)
            return {'error': str(e)}
    
    def save_results(self, output_dir: str = "reports/benchmarks"):
        """Сохранение результатов бенчмарка"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = output_path / f"benchmark_{timestamp}.json"
        
        results_with_metadata = {
            'timestamp': timestamp,
            'models_tested': list(self.model_paths.keys()),
            'hardware': self.benchmark_hardware(),
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        model_logger.info(f"Benchmark results saved to: {filepath}")
        
        # Генерация краткого отчета
        self._generate_summary_report(filepath)
        
        return filepath
    
    def _generate_summary_report(self, results_path: Path):
        """Генерация краткого отчета в Markdown"""
        summary_path = results_path.parent / f"{results_path.stem}_summary.md"
        
        with open(summary_path, 'w') as f:
            f.write("# Model Benchmark Summary\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Models Tested\n")
            for model_name in self.model_paths.keys():
                f.write(f"- {model_name}\n")
            
            f.write("\n## Performance Comparison\n")
            
            if 'comparison' in self.results:
                f.write("| Model | Throughput vs Baseline | Latency vs Baseline | Recommendation |\n")
                f.write("|-------|------------------------|---------------------|----------------|\n")
                
                for model_name, comparison in self.results['comparison'].items():
                    if isinstance(comparison, dict):
                        throughput = comparison.get('throughput_vs_baseline', 1)
                        latency = comparison.get('latency_vs_baseline', 1)
                        recommendation = comparison.get('recommendation', '')
                        
                        f.write(f"| {model_name} | {throughput:.2f}x | {latency:.2f}x | {recommendation} |\n")
            
            f.write("\n## Hardware Information\n")
            hardware = self.benchmark_hardware()
            
            if 'cpu' in hardware:
                cpu = hardware['cpu']
                f.write(f"- CPU: {cpu.get('physical_cores')} cores, {cpu.get('frequency_mhz', 'N/A')} MHz\n")
            
            if 'memory' in hardware:
                memory = hardware['memory']
                f.write(f"- Memory: {memory.get('total_gb', 0):.1f} GB total\n")
        
        model_logger.info(f"Summary report saved to: {summary_path}")

class OptimizationBenchmark(ModelBenchmark):
    """Бенчмарк для сравнения оптимизированных моделей"""
    
    def __init__(self, original_model_path: str):
        self.original_model_path = original_model_path
        self.optimizer = ModelOptimizer(original_model_path, input_size=20)
        
        # Пути к оптимизированным моделям
        model_paths = {
            'original': original_model_path
        }
        
        super().__init__(model_paths)
    
    def create_optimized_models(self):
        """Создание оптимизированных версий модели"""
        model_logger.info("Creating optimized model variants")
        
        try:
            # Загрузка оригинальной модели
            import torch
            from src.ml_pipeline.training.train_model import CreditScoringNN
            
            input_size = 20  # Пример
            original_model = CreditScoringNN(input_size)
            original_model.load_state_dict(torch.load(self.original_model_path))
            
            # 1. Pruned модель
            pruned_model = self.optimizer.apply_pruning(original_model, pruning_rate=0.2)
            pruned_path = self.original_model_path.replace('.pth', '_pruned.pth')
            torch.save(pruned_model.state_dict(), pruned_path)
            self.model_paths['pruned'] = pruned_path
            
            # 2. Quantized модель
            quantized_model = self.optimizer.dynamic_quantization(original_model)
            quantized_path = self.original_model_path.replace('.pth', '_quantized.pth')
            torch.jit.save(torch.jit.script(quantized_model), quantized_path)
            self.model_paths['quantized'] = quantized_path
            
            # 3. ONNX модель
            from src.ml_pipeline.training.onnx_conversion import ModelConverter
            converter = ModelConverter()
            onnx_path = converter.convert_to_onnx(original_model, input_size)
            self.model_paths['onnx'] = onnx_path
            
            # 4. Quantized ONNX
            quantized_onnx = self.optimizer.quantize_onnx_model(onnx_path)
            self.model_paths['onnx_quantized'] = quantized_onnx
            
            model_logger.info(f"Created {len(self.model_paths)} model variants")
            
        except Exception as e:
            model_logger.error(f"Failed to create optimized models: {e}", exc_info=True)
            raise
    
    def compare_optimizations(self):
        """Сравнение разных методов оптимизации"""
        # Создание оптимизированных моделей
        self.create_optimized_models()
        
        # Запуск бенчмарка
        results = self.benchmark_models()
        
        # Анализ эффективности оптимизаций
        optimization_analysis = self._analyze_optimizations(results)
        
        # Генерация рекомендаций
        recommendations = self._generate_optimization_recommendations(
            optimization_analysis
        )
        
        return {
            'results': results,
            'analysis': optimization_analysis,
            'recommendations': recommendations
        }
    
    def _analyze_optimizations(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ эффективности разных методов оптимизации"""
        analysis = {}
        
        original_results = benchmark_results.get('original', {}).get('summary', {})
        
        for model_name, results in benchmark_results.items():
            if model_name == 'comparison' or 'error' in results:
                continue
            
            model_summary = results.get('summary', {})
            
            if model_summary and original_results:
                analysis[model_name] = {
                    'throughput_improvement': (
                        model_summary.get('best_throughput_rps', 0) / 
                        max(original_results.get('best_throughput_rps', 1), 1)
                    ),
                    'latency_improvement': (
                        original_results.get('best_latency_ms', 1) / 
                        max(model_summary.get('best_latency_ms', 1), 1)
                    ),
                    'optimal_batch_size': model_summary.get('optimal_batch_size'),
                    'memory_efficiency': self._calculate_memory_efficiency(model_name)
                }
        
        return analysis
    
    def _calculate_memory_efficiency(self, model_name: str) -> float:
        """Расчет эффективности использования памяти"""
        try:
            model_path = self.model_paths[model_name]
            model_size = Path(model_path).stat().st_size / 1024  # KB
            
            original_size = Path(self.original_model_path).stat().st_size / 1024
            
            return original_size / max(model_size, 1)
            
        except:
            return 1.0
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        recommendations = []
        
        best_throughput = max(
            (data.get('throughput_improvement', 1), name)
            for name, data in analysis.items()
        )
        
        best_latency = max(
            (data.get('latency_improvement', 1), name)
            for name, data in analysis.items()
        )
        
        best_memory = max(
            (data.get('memory_efficiency', 1), name)
            for name, data in analysis.items()
        )
        
        if best_throughput[0] > 1.2:
            recommendations.append(
                f"Для максимальной пропускной способности используйте {best_throughput[1]} "
                f"(ускорение в {best_throughput[0]:.2f}x)"
            )
        
        if best_latency[0] > 1.2:
            recommendations.append(
                f"Для минимальной задержки используйте {best_latency[1]} "
                f"(ускорение в {best_latency[0]:.2f}x)"
            )
        
        if best_memory[0] > 1.5:
            recommendations.append(
                f"Для экономии памяти используйте {best_memory[1]} "
                f"(сжатие в {best_memory[0]:.2f}x)"
            )
        
        if not recommendations:
            recommendations.append(
                "Оптимизации не показали значительного улучшения. "
                "Рекомендуется использовать оригинальную модель."
            )
        
        return recommendations

if __name__ == "__main__":
    # Пример использования
    benchmark = ModelBenchmark({
        'credit_scoring_onnx': 'models/credit_scoring.onnx',
        'credit_scoring_quantized': 'models/credit_scoring_quantized.onnx'
    })
    
    # Запуск бенчмарка
    results = benchmark.benchmark_models()
    
    # Стресс-тест
    stress_results = benchmark.run_stress_test('credit_scoring_onnx', duration=60)
    
    print("Benchmark completed")
    print(f"Best throughput: {results.get('comparison', {})}")