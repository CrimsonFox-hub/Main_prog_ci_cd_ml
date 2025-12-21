"""
Нагрузочное тестирование API
"""
import time
import threading
import requests
import statistics
import json
from pathlib import Path
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm

class LoadTester:
    def __init__(self, base_url, endpoint="/api/v1/predict"):
        self.base_url = base_url.rstrip('/')
        self.endpoint = endpoint
        self.results = []
        
    def generate_sample_request(self):
        """Генерация тестового запроса"""
        # Пример данных для кредитного скоринга
        return {
            "features": {
                "age": np.random.randint(18, 70),
                "income": np.random.randint(20000, 150000),
                "credit_amount": np.random.randint(1000, 50000),
                "loan_duration": np.random.randint(6, 60),
                "payment_to_income": round(np.random.uniform(0.1, 0.5), 2),
                "existing_credits": np.random.randint(0, 5),
                "dependents": np.random.randint(0, 5),
                "residence_since": np.random.randint(0, 20),
                "installment_rate": round(np.random.uniform(1.0, 4.0), 2),
                "credit_history": np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34']),
                "purpose": np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44']),
                "savings": np.random.choice(['A61', 'A62', 'A63', 'A64'])
            }
        }
    
    def make_request(self, request_id):
        """Отправка одного запроса"""
        start_time = time.time()
        status_code = 0
        error = None
        
        try:
            response = requests.post(
                f"{self.base_url}{self.endpoint}",
                json=self.generate_sample_request(),
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            status_code = response.status_code
            if response.status_code != 200:
                error = f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            error = str(e)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        return {
            'request_id': request_id,
            'timestamp': start_time,
            'latency_ms': latency,
            'status_code': status_code,
            'error': error,
            'success': error is None and status_code == 200
        }
    
    def run_test(self, concurrent_users, duration_sec, request_rate=None):
        """Запуск нагрузочного теста"""
        print(f"\nЗапуск теста: {concurrent_users} пользователей, {duration_sec} секунд")
        
        self.results = []
        start_time = time.time()
        request_id = 0
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            if request_rate:
                requests_per_second = request_rate
                interval = 1.0 / requests_per_second
            else:
                interval = 0
            
            pbar = tqdm(total=duration_sec, desc="Прогресс теста")
            
            while time.time() - start_time < duration_sec:
                for _ in range(concurrent_users):
                    future = executor.submit(self.make_request, request_id)
                    futures.append(future)
                    request_id += 1
                
                if interval > 0:
                    time.sleep(interval)
                
                elapsed = time.time() - start_time
                pbar.update(int(elapsed) - pbar.n)
            
            pbar.close()

            for future in as_completed(futures):
                self.results.append(future.result())
        
        return self.analyze_results(concurrent_users, duration_sec)
    
    def analyze_results(self, concurrent_users, duration_sec):
        """Анализ результатов теста"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        total_requests = len(df)
        successful_requests = df['success'].sum()
        failed_requests = total_requests - successful_requests
        
        if successful_requests > 0:
            successful_latencies = df[df['success']]['latency_ms'].tolist()
            
            metrics = {
                'concurrent_users': concurrent_users,
                'duration_sec': duration_sec,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': successful_requests / total_requests,
                'throughput_rps': total_requests / duration_sec,
                'latency_avg_ms': statistics.mean(successful_latencies),
                'latency_p50_ms': statistics.median(successful_latencies),
                'latency_p95_ms': np.percentile(successful_latencies, 95),
                'latency_p99_ms': np.percentile(successful_latencies, 99),
                'latency_min_ms': min(successful_latencies),
                'latency_max_ms': max(successful_latencies),
                'timestamp': datetime.now().isoformat(),
                'endpoint': self.endpoint
            }
        else:
            metrics = {
                'concurrent_users': concurrent_users,
                'duration_sec': duration_sec,
                'total_requests': total_requests,
                'successful_requests': 0,
                'failed_requests': total_requests,
                'success_rate': 0,
                'throughput_rps': 0,
                'error': 'Все запросы завершились ошибкой'
            }
        
        # Вывод результатов
        print(f"\nРезультаты теста ({concurrent_users} пользователей):")
        print(f"  Всего запросов: {metrics['total_requests']}")
        print(f"  Успешных: {metrics['successful_requests']} ({metrics['success_rate']:.1%})")
        print(f"  Пропускная способность: {metrics['throughput_rps']:.1f} запр/сек")
        
        if 'latency_avg_ms' in metrics:
            print(f"  Задержка (средняя): {metrics['latency_avg_ms']:.1f} мс")
            print(f"  Задержка (P95): {metrics['latency_p95_ms']:.1f} мс")
            print(f"  Задержка (P99): {metrics['latency_p99_ms']:.1f} мс")
        
        return metrics

def run_load_test_scenarios(args):
    """Запуск нескольких сценариев нагрузочного тестирования"""
    tester = LoadTester(args.base_url, args.endpoint)
    all_results = []

    scenarios = [
        {'users': 10, 'duration': 30, 'rate': 50},
        {'users': 50, 'duration': 60, 'rate': 100},
        {'users': 100, 'duration': 90, 'rate': 200},
        {'users': 200, 'duration': 120, 'rate': 500},
    ]
    
    for scenario in scenarios:
        print(f"\nСценарий: {scenario['users']} пользователей, "
              f"{scenario['duration']} секунд, {scenario['rate']} запр/сек")
        
        result = tester.run_test(
            concurrent_users=scenario['users'],
            duration_sec=scenario['duration'],
            request_rate=scenario['rate']
        )
        
        if result:
            all_results.append(result)
        
        # Пауза между тестами
        if scenario != scenarios[-1]:
            print("Пауза 10 секунд перед следующим тестом...")
            time.sleep(10)
    
    # Анализ всех результатов
    if all_results:
        df_results = pd.DataFrame(all_results)
        optimal_idx = df_results[
            (df_results['success_rate'] > 0.95) & 
            (df_results['latency_p95_ms'] < 500)
        ]['throughput_rps'].idxmax() if 'latency_p95_ms' in df_results.columns else 0
        
        if not pd.isna(optimal_idx):
            optimal = df_results.loc[optimal_idx]
            print(f"\nОптимальная конфигурация:")
            print(f"  Пользователей: {optimal['concurrent_users']}")
            print(f"  Пропускная способность: {optimal['throughput_rps']:.1f} запр/сек")
            print(f"  Задержка P95: {optimal['latency_p95_ms']:.1f} мс")
            print(f"  Успешность: {optimal['success_rate']:.1%}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_path = output_dir / f"load_test_detailed_{timestamp}.json"
        with open(detailed_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        summary_path = output_dir / f"load_test_summary_{timestamp}.csv"
        df_results.to_csv(summary_path, index=False)
        recommendations = {
            'optimal_configuration': optimal.to_dict() if not pd.isna(optimal_idx) else {},
            'max_safe_load': df_results[df_results['success_rate'] > 0.95]['throughput_rps'].max(),
            'recommended_concurrent_users': int(optimal['concurrent_users'] * 0.8) if not pd.isna(optimal_idx) else 50,
            'sla_violations': len(df_results[df_results['latency_p95_ms'] > 500]) if 'latency_p95_ms' in df_results.columns else 0,
            'test_timestamp': timestamp,
            'base_url': args.base_url
        }
        
        recommendations_path = output_dir / f"load_test_recommendations_{timestamp}.json"
        with open(recommendations_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"\nРезультаты сохранены в: {output_dir}")
        print(f"  Подробные результаты: {detailed_path}")
        print(f"  Сводный отчет: {summary_path}")
        print(f"  Рекомендации: {recommendations_path}")

def main():
    parser = argparse.ArgumentParser(description='Нагрузочное тестирование API')
    parser.add_argument('--base-url', default='http://localhost:8000',
                       help='Базовый URL API')
    parser.add_argument('--endpoint', default='/api/v1/predict',
                       help='Эндпоинт для тестирования')
    parser.add_argument('--output-dir', default='reports/load_tests',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    run_load_test_scenarios(args)

if __name__ == "__main__":
    main()