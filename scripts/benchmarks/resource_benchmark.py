"""
Бенчмарк ресурсов для разных типов инстансов (CPU/GPU)
Этап 1: Определение оптимальной конфигурации ресурсов
"""
import time
import numpy as np
import pandas as pd
import psutil
import GPUtil
import json
from pathlib import Path
import argparse
from datetime import datetime
import multiprocessing
import threading
import queue
import sys
from typing import Dict, List, Tuple

class ResourceBenchmark:
    def __init__(self):
        self.results = {}
        self.cpu_count = multiprocessing.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        
    def measure_cpu_performance(self, n_threads: int = None, duration: int = 5) -> Dict:
        """Измерение производительности CPU"""
        print(f"Тестирование CPU (потоков: {n_threads or 'auto'})...")
        
        if n_threads is None:
            n_threads = self.cpu_count
        
        # Создаем задачу для нагрузки на CPU
        def cpu_intensive_task(n_iterations: int):
            result = 0
            for i in range(n_iterations):
                result += (i * i) / 3.14159
            return result
        
        # Тестируем однопоточную производительность
        start_time = time.perf_counter()
        result = cpu_intensive_task(1000000)
        single_thread_time = time.perf_counter() - start_time
        
        # Тестируем многопоточную производительность
        threads = []
        results_queue = queue.Queue()
        
        def worker(task_id, iterations):
            start = time.perf_counter()
            result = cpu_intensive_task(iterations)
            end = time.perf_counter()
            results_queue.put((task_id, result, end - start))
        
        start_time = time.perf_counter()
        for i in range(n_threads):
            thread = threading.Thread(
                target=worker,
                args=(i, 1000000 // n_threads)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        multi_thread_time = time.perf_counter() - start_time
        
        # Собираем результаты
        cpu_metrics = {
            'cpu_count': self.cpu_count,
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'tested_threads': n_threads,
            'single_thread_time_sec': single_thread_time,
            'multi_thread_time_sec': multi_thread_time,
            'speedup_factor': single_thread_time / multi_thread_time if multi_thread_time > 0 else 0,
            'cpu_utilization_percent': psutil.cpu_percent(interval=1),
            'cpu_times': dict(psutil.cpu_times()._asdict()),
            'timestamp': datetime.now().isoformat()
        }
        
        return cpu_metrics
    
    def measure_memory_bandwidth(self, size_mb: int = 100) -> Dict:
        """Измерение пропускной способности памяти"""
        print(f"Тестирование памяти ({size_mb} MB)...")
        
        # Создаем большой массив для тестирования
        size_bytes = size_mb * 1024 * 1024
        element_size = 8  # float64
        n_elements = size_bytes // element_size
        
        # Тест записи
        start_time = time.perf_counter()
        array = np.zeros(n_elements, dtype=np.float64)
        write_time = time.perf_counter() - start_time
        
        # Тест чтения
        start_time = time.perf_counter()
        _ = array.sum()
        read_time = time.perf_counter() - start_time
        
        # Тест копирования
        start_time = time.perf_counter()
        copy_array = array.copy()
        copy_time = time.perf_counter() - start_time
        
        # Рассчитываем пропускную способность
        write_bandwidth = size_bytes / write_time / (1024**3)  # GB/s
        read_bandwidth = size_bytes / read_time / (1024**3)    # GB/s
        copy_bandwidth = size_bytes / copy_time / (1024**3)    # GB/s
        
        memory_metrics = {
            'total_memory_gb': self.total_memory / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'test_size_mb': size_mb,
            'write_bandwidth_gbs': write_bandwidth,
            'read_bandwidth_gbs': read_bandwidth,
            'copy_bandwidth_gbs': copy_bandwidth,
            'memory_used_percent': psutil.virtual_memory().percent,
            'timestamp': datetime.now().isoformat()
        }
        
        return memory_metrics
    
    def measure_disk_performance(self, test_dir: str = '/tmp', file_size_mb: int = 100) -> Dict:
        """Измерение производительности диска"""
        print(f"Тестирование диска ({file_size_mb} MB)...")
        
        test_file = Path(test_dir) / 'benchmark_test.bin'
        test_data = os.urandom(file_size_mb * 1024 * 1024)
        
        # Тест записи
        start_time = time.perf_counter()
        with open(test_file, 'wb') as f:
            f.write(test_data)
        write_time = time.perf_counter() - start_time
        
        # Синхронизация (если нужно)
        os.sync()
        
        # Тест чтения
        start_time = time.perf_counter()
        with open(test_file, 'rb') as f:
            _ = f.read()
        read_time = time.perf_counter() - start_time
        
        # Очистка
        test_file.unlink()
        
        # Рассчитываем скорость
        file_size_bytes = file_size_mb * 1024 * 1024
        write_speed = file_size_bytes / write_time / (1024**2)  # MB/s
        read_speed = file_size_bytes / read_time / (1024**2)    # MB/s
        
        disk_metrics = {
            'test_file_size_mb': file_size_mb,
            'write_speed_mbs': write_speed,
            'read_speed_mbs': read_speed,
            'disk_usage_percent': psutil.disk_usage(test_dir).percent,
            'test_directory': test_dir,
            'timestamp': datetime.now().isoformat()
        }
        
        return disk_metrics
    
    def measure_gpu_performance(self) -> Dict:
        """Измерение производительности GPU (если доступно)"""
        print("Тестирование GPU...")
        
        gpu_metrics = {'available': False}
        
        try:
            import torch
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu = gpus[0]
                gpu_metrics.update({
                    'available': True,
                    'gpu_name': gpu.name,
                    'gpu_memory_total_mb': gpu.memoryTotal,
                    'gpu_memory_used_mb': gpu.memoryUsed,
                    'gpu_memory_free_mb': gpu.memoryFree,
                    'gpu_load_percent': gpu.load * 100,
                    'gpu_temperature_c': gpu.temperature
                })
                
                # Тест производительности GPU с PyTorch
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    
                    # Тест матричного умножения
                    size = 4096
                    a = torch.randn(size, size, device=device)
                    b = torch.randn(size, size, device=device)
                    
                    # Прогрев
                    for _ in range(10):
                        _ = torch.matmul(a, b)
                    
                    # Измерение
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    
                    for _ in range(100):
                        _ = torch.matmul(a, b)
                    
                    torch.cuda.synchronize()
                    gpu_time = time.perf_counter() - start_time
                    
                    gpu_metrics['matrix_multiply_time_sec'] = gpu_time
                    gpu_metrics['gflops'] = (2 * size**3 * 100) / (gpu_time * 1e9)
            
        except ImportError:
            print("GPU библиотеки не установлены")
        except Exception as e:
            print(f"Ошибка тестирования GPU: {e}")
        
        gpu_metrics['timestamp'] = datetime.now().isoformat()
        return gpu_metrics
    
    def measure_model_inference_resources(self, model_path: str, input_shape: Tuple) -> Dict:
        """Измерение ресурсов для инференса модели"""
        print(f"Тестирование инференса модели {model_path}...")
        
        try:
            import joblib
            import onnxruntime as ort
            
            # Загрузка модели
            if model_path.endswith('.pkl'):
                model = joblib.load(model_path)
                model_type = 'sklearn'
            elif model_path.endswith('.onnx'):
                session = ort.InferenceSession(model_path)
                model_type = 'onnx'
            else:
                raise ValueError(f"Неподдерживаемый формат модели: {model_path}")
            
            # Генерация тестовых данных
            np.random.seed(42)
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Измерение CPU использования
            process = psutil.Process()
            cpu_percent_before = process.cpu_percent(interval=None)
            memory_before = process.memory_info().rss
            
            # Инференс
            start_time = time.perf_counter()
            
            if model_type == 'sklearn':
                predictions = model.predict(test_input)
            else:
                input_name = session.get_inputs()[0].name
                predictions = session.run(None, {input_name: test_input})
            
            inference_time = time.perf_counter() - start_time
            
            cpu_percent_after = process.cpu_percent(interval=None)
            memory_after = process.memory_info().rss
            
            inference_metrics = {
                'model_type': model_type,
                'model_path': model_path,
                'input_shape': input_shape,
                'batch_size': input_shape[0],
                'inference_time_sec': inference_time,
                'inference_per_second': input_shape[0] / inference_time,
                'cpu_usage_percent': cpu_percent_after,
                'cpu_usage_delta': cpu_percent_after - cpu_percent_before,
                'memory_usage_mb': memory_after / (1024**2),
                'memory_delta_mb': (memory_after - memory_before) / (1024**2),
                'timestamp': datetime.now().isoformat()
            }
            
            return inference_metrics
            
        except Exception as e:
            print(f"Ошибка тестирования инференса: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_benchmark(self, output_dir: str = 'reports/benchmarks') -> Dict:
        """Запуск комплексного бенчмарка"""
        print("=" * 60)
        print("КОМПЛЕКСНЫЙ БЕНЧМАРК РЕСУРСОВ")
        print("=" * 60)
        
        all_results = {}
        
        # 1. Производительность CPU
        print("\n1. Тестирование CPU...")
        cpu_results = self.measure_cpu_performance()
        all_results['cpu'] = cpu_results
        
        # 2. Пропускная способность памяти
        print("\n2. Тестирование памяти...")
        memory_results = self.measure_memory_bandwidth()
        all_results['memory'] = memory_results
        
        # 3. Производительность диска
        print("\n3. Тестирование диска...")
        disk_results = self.measure_disk_performance()
        all_results['disk'] = disk_results
        
        # 4. Производительность GPU (если есть)
        print("\n4. Тестирование GPU...")
        gpu_results = self.measure_gpu_performance()
        all_results['gpu'] = gpu_results
        
        # 5. Сводная информация о системе
        print("\n5. Сбор системной информации...")
        system_info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'timestamp': datetime.now().isoformat()
        }
        all_results['system'] = system_info
        
        # Сохранение результатов
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_path / f"resource_benchmark_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Генерация рекомендаций
        recommendations = self.generate_recommendations(all_results)
        all_results['recommendations'] = recommendations
        
        # Вывод сводки
        self.print_summary(all_results)
        
        print(f"\nПолный отчет сохранен в: {report_path}")
        return all_results
    
    def generate_recommendations(self, results: Dict) -> Dict:
        """Генерация рекомендаций по конфигурации"""
        recs = {
            'optimal_instance_type': 'unknown',
            'min_cpu_cores': 2,
            'min_memory_gb': 4,
            'recommended_cpu_cores': 4,
            'recommended_memory_gb': 8,
            'use_gpu': False,
            'storage_type': 'ssd',
            'notes': []
        }
        
        # Анализ CPU
        if 'cpu' in results:
            cpu_info = results['cpu']
            cpu_cores = cpu_info['cpu_count']
            
            if cpu_cores >= 8:
                recs['optimal_instance_type'] = 'cpu-optimized'
                recs['recommended_cpu_cores'] = 4
            elif cpu_cores >= 4:
                recs['optimal_instance_type'] = 'general-purpose'
                recs['recommended_cpu_cores'] = 2
            else:
                recs['optimal_instance_type'] = 'burstable'
                recs['recommended_cpu_cores'] = 1
            
            if cpu_info.get('speedup_factor', 0) < 1.5:
                recs['notes'].append('Низкий прирост от многопоточности - рассмотрите single-core оптимизацию')
        
        # Анализ памяти
        if 'memory' in results:
            mem_info = results['memory']
            total_mem_gb = mem_info['total_memory_gb']
            
            if total_mem_gb >= 16:
                recs['recommended_memory_gb'] = 8
                recs['min_memory_gb'] = 4
            elif total_mem_gb >= 8:
                recs['recommended_memory_gb'] = 4
                recs['min_memory_gb'] = 2
            else:
                recs['recommended_memory_gb'] = 2
                recs['min_memory_gb'] = 1
            
            if mem_info.get('read_bandwidth_gbs', 0) < 10:
                recs['notes'].append('Низкая пропускная способность памяти - возможны узкие места')
        
        # Анализ GPU
        if 'gpu' in results and results['gpu'].get('available', False):
            gpu_info = results['gpu']
            if gpu_info.get('gpu_memory_total_mb', 0) >= 4000:
                recs['use_gpu'] = True
                recs['optimal_instance_type'] = 'gpu-optimized'
                recs['notes'].append('Доступен GPU - используйте для тренировки и инференса')
        
        # Анализ диска
        if 'disk' in results:
            disk_info = results['disk']
            if disk_info.get('write_speed_mbs', 0) > 200:
                recs['storage_type'] = 'nvme-ssd'
            elif disk_info.get('write_speed_mbs', 0) > 100:
                recs['storage_type'] = 'ssd'
            else:
                recs['storage_type'] = 'hdd'
                recs['notes'].append('Медленный диск - рассмотрите upgrade до SSD')
        
        return recs
    
    def print_summary(self, results: Dict):
        """Вывод сводки результатов"""
        # CPU
        if 'cpu' in results:
            cpu = results['cpu']
            print(f"\nCPU:")
            print(f"  Ядер: {cpu['cpu_count']}")
            print(f"  Частота: {cpu.get('cpu_freq', 'N/A')} MHz")
            print(f"  Ускорение многопоточности: {cpu.get('speedup_factor', 0):.2f}x")
            print(f"  Загрузка: {cpu.get('cpu_utilization_percent', 0):.1f}%")
        
        # Память
        if 'memory' in results:
            mem = results['memory']
            print(f"\nПамять:")
            print(f"  Всего: {mem['total_memory_gb']:.1f} GB")
            print(f"  Доступно: {mem['available_memory_gb']:.1f} GB")
            print(f"  Пропускная способность: {mem.get('read_bandwidth_gbs', 0):.1f} GB/s")
        
        # Диск
        if 'disk' in results:
            disk = results['disk']
            print(f"\nДиск:")
            print(f"  Скорость чтения: {disk['read_speed_mbs']:.1f} MB/s")
            print(f"  Скорость записи: {disk['write_speed_mbs']:.1f} MB/s")
            print(f"  Тип: {disk.get('storage_type', 'unknown')}")
        
        # GPU
        if 'gpu' in results and results['gpu'].get('available'):
            gpu = results['gpu']
            print(f"\nGPU:")
            print(f"  Модель: {gpu.get('gpu_name', 'unknown')}")
            print(f"  Память: {gpu.get('gpu_memory_total_mb', 0)} MB")
            print(f"  Загрузка: {gpu.get('gpu_load_percent', 0):.1f}%")
        
        # Рекомендации
        if 'recommendations' in results:
            recs = results['recommendations']
            print(f"\nРЕКОМЕНДАЦИИ:")
            print(f"  Оптимальный тип инстанса: {recs['optimal_instance_type']}")
            print(f"  Рекомендуемые CPU ядра: {recs['recommended_cpu_cores']}")
            print(f"  Рекомендуемая память: {recs['recommended_memory_gb']} GB")
            print(f"  Использовать GPU: {'Да' if recs['use_gpu'] else 'Нет'}")
            print(f"  Тип хранилища: {recs['storage_type']}")
            
            if recs['notes']:
                print(f"\nЗаметки:")
                for note in recs['notes']:
                    print(f"  - {note}")

def main():
    parser = argparse.ArgumentParser(description='Бенчмарк ресурсов системы')
    parser.add_argument('--output-dir', default='reports/benchmarks',
                       help='Директория для сохранения отчетов')
    parser.add_argument('--test-model', help='Путь к модели для тестирования инференса')
    parser.add_argument('--input-shape', default='1,20',
                       help='Форма входных данных для модели (например: 1,20)')
    
    args = parser.parse_args()
    
    benchmark = ResourceBenchmark()
    results = benchmark.run_comprehensive_benchmark(args.output_dir)
    
    # Дополнительное тестирование инференса если указана модель
    if args.test_model:
        try:
            input_shape = tuple(map(int, args.input_shape.split(',')))
            inference_results = benchmark.measure_model_inference_resources(
                args.test_model, input_shape
            )
            
            if 'error' not in inference_results:
                print(f"\nРезультаты инференса модели:")
                print(f"  Время: {inference_results['inference_time_sec']:.4f} сек")
                print(f"  Скорость: {inference_results['inference_per_second']:.1f} запр/сек")
                print(f"  Использование памяти: {inference_results['memory_usage_mb']:.1f} MB")
        
        except Exception as e:
            print(f"Ошибка тестирования инференса: {e}")
    
    return 0

if __name__ == "__main__":
    import platform
    import os
    
    if __name__ == "__main__":
        main()