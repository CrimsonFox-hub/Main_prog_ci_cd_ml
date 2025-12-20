"""
Сравнение производительности оригинальной модели и ONNX версии
Этап 1: Конвертация в ONNX и сравнение производительности
"""
import time
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from sklearn.metrics import accuracy_score, roc_auc_score
import psutil
import json
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_models(model_path_pkl, model_path_onnx):
    """Загрузка моделей"""
    print("Загрузка моделей...")
    
    # Загрузка оригинальной модели
    try:
        original_model = joblib.load(model_path_pkl)
        print(f"Оригинальная модель загружена: {model_path_pkl}")
    except Exception as e:
        print(f"Ошибка загрузки оригинальной модели: {e}")
        return None, None
    
    # Загрузка ONNX модели
    try:
        onnx_session = ort.InferenceSession(model_path_onnx)
        print(f"ONNX модель загружена: {model_path_onnx}")
    except Exception as e:
        print(f"Ошибка загрузки ONNX модели: {e}")
        return original_model, None
    
    return original_model, onnx_session

def prepare_test_data(data_path, n_samples=10000):
    """Подготовка тестовых данных"""
    print(f"Загрузка тестовых данных из {data_path}...")
    
    try:
        data = pd.read_csv(data_path)
        
        # Предполагаем, что последняя колонка - target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Если нужно больше данных, генерируем
        if len(X) < n_samples:
            print(f"Генерация дополнительных данных ({n_samples} образцов)...")
            # Простая генерация на основе существующих данных
            X = np.random.randn(n_samples, X.shape[1])
            y = np.random.randint(0, 2, n_samples)
        else:
            # Берем первые n_samples
            X = X[:n_samples]
            y = y[:n_samples]
        
        print(f"Тестовые данные: {X.shape[0]} образцов, {X.shape[1]} признаков")
        return X, y
        
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        # Генерация синтетических данных
        print("Генерация синтетических данных...")
        X = np.random.randn(n_samples, 20)
        y = np.random.randint(0, 2, n_samples)
        return X, y

def benchmark_original_model(model, X, y, n_runs=100):
    """Бенчмарк оригинальной модели"""
    print("\nБенчмарк оригинальной модели...")
    
    times = []
    predictions = []
    
    # Прогреваем кэш
    for _ in range(10):
        model.predict(X[:10])
    
    # Основной бенчмарк
    for i in range(n_runs):
        start_time = time.perf_counter()
        pred = model.predict(X)
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # мс
        
        if i == 0:  # Сохраняем предсказания для оценки качества
            predictions = pred
    
    # Метрики
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = len(X) / (avg_time / 1000)  # запросов в секунду
    
    # Качество модели
    if len(predictions) > 0:
        accuracy = accuracy_score(y, predictions)
        try:
            proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else predictions
            roc_auc = roc_auc_score(y, proba)
        except:
            roc_auc = None
    else:
        accuracy = None
        roc_auc = None
    
    # Использование памяти
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return {
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'throughput_rps': throughput,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'memory_usage_mb': memory_mb,
        'n_runs': n_runs,
        'batch_size': len(X)
    }

def benchmark_onnx_model(session, X, y, n_runs=100):
    """Бенчмарк ONNX модели"""
    print("\nБенчмарк ONNX модели...")
    
    times = []
    predictions = []
    
    # Получаем имя входного тензора
    input_name = session.get_inputs()[0].name
    
    # Прогреваем кэш
    for _ in range(10):
        session.run(None, {input_name: X[:10].astype(np.float32)})
    
    # Основной бенчмарк
    for i in range(n_runs):
        start_time = time.perf_counter()
        result = session.run(None, {input_name: X.astype(np.float32)})
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # мс
        
        if i == 0:  # Сохраняем предсказания
            predictions = result[0].flatten()
    
    # Метрики
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = len(X) / (avg_time / 1000)
    
    # Качество модели (бинарная классификация)
    if len(predictions) > 0:
        predictions_binary = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(y, predictions_binary)
        roc_auc = roc_auc_score(y, predictions)
    else:
        accuracy = None
        roc_auc = None
    
    # Использование памяти
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return {
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'throughput_rps': throughput,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'memory_usage_mb': memory_mb,
        'n_runs': n_runs,
        'batch_size': len(X)
    }

def run_benchmark(args):
    """Запуск бенчмарка"""
    print("=" * 60)
    print("Бенчмарк моделей: оригинальная vs ONNX")
    print("=" * 60)
    
    # Загрузка моделей
    original_model, onnx_session = load_models(args.model_pkl, args.model_onnx)
    
    if original_model is None and onnx_session is None:
        print("Не удалось загрузить модели")
        return
    
    # Подготовка данных
    X, y = prepare_test_data(args.data_path, args.n_samples)
    
    results = {}
    
    # Бенчмарк оригинальной модели
    if original_model is not None:
        results['original'] = benchmark_original_model(original_model, X, y, args.n_runs)
    
    # Бенчмарк ONNX модели
    if onnx_session is not None:
        results['onnx'] = benchmark_onnx_model(onnx_session, X, y, args.n_runs)
    
    # Сравнение результатов
    if 'original' in results and 'onnx' in results:
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 60)
        
        orig = results['original']
        onnx = results['onnx']
        
        # Время инференса
        speedup = orig['avg_inference_time_ms'] / onnx['avg_inference_time_ms']
        print(f"\nСкорость инференса:")
        print(f"  Оригинальная: {orig['avg_inference_time_ms']:.2f} ± {orig['std_inference_time_ms']:.2f} мс")
        print(f"  ONNX: {onnx['avg_inference_time_ms']:.2f} ± {onnx['std_inference_time_ms']:.2f} мс")
        print(f"  Ускорение: {speedup:.2f}x")
        
        # Пропускная способность
        print(f"\nПропускная способность:")
        print(f"  Оригинальная: {orig['throughput_rps']:.0f} запросов/сек")
        print(f"  ONNX: {onnx['throughput_rps']:.0f} запросов/сек")
        
        # Качество
        if orig['accuracy'] is not None and onnx['accuracy'] is not None:
            print(f"\nКачество модели:")
            print(f"  Оригинальная - Accuracy: {orig['accuracy']:.4f}, ROC-AUC: {orig['roc_auc']:.4f}")
            print(f"  ONNX - Accuracy: {onnx['accuracy']:.4f}, ROC-AUC: {onnx['roc_auc']:.4f}")
        
        # Память
        print(f"\nИспользование памяти:")
        print(f"  Оригинальная: {orig['memory_usage_mb']:.1f} MB")
        print(f"  ONNX: {onnx['memory_usage_mb']:.1f} MB")
    
    # Сохранение результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nРезультаты сохранены в: {output_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Бенчмарк моделей: оригинальная vs ONNX')
    parser.add_argument('--model-pkl', default='models/trained/credit_scoring_model.pkl',
                       help='Путь к оригинальной модели (.pkl)')
    parser.add_argument('--model-onnx', default='models/trained/credit_scoring_model.onnx',
                       help='Путь к ONNX модели (.onnx)')
    parser.add_argument('--data-path', default='data/processed/test.csv',
                       help='Путь к тестовым данным')
    parser.add_argument('--output-dir', default='reports/benchmarks',
                       help='Директория для сохранения результатов')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Количество тестовых образцов')
    parser.add_argument('--n-runs', type=int, default=100,
                       help='Количество запусков для каждого теста')
    
    args = parser.parse_args()
    run_benchmark(args)

if __name__ == "__main__":
    main()