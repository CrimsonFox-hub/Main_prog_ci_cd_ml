"""
Интеграционные тесты API endpoints
"""
import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Добавление пути к исходному коду
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.app import app
from src.ml_pipeline.inference.predictor import Predictor

class TestAPIEndpoints:
    """Тесты для API endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Настройка тестового окружения"""
        # Создание временной директории для тестов
        self.test_dir = tmp_path
        self.model_path = self.test_dir / "test_model.onnx"
        
        # Создание тестовой модели
        self._create_test_model()
        
        # Создание тестового клиента
        self.client = TestClient(app)
        
        # Переопределение конфигурации приложения для тестов
        app.state.config = {
            'model': {
                'path': str(self.model_path),
                'input_shape': (1, 20),
                'output_shape': (1, 1)
            }
        }
        
        yield
        
        # Очистка после тестов
        if hasattr(self, 'test_dir'):
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_model(self):
        """Создание тестовой ONNX модели"""
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
        
        # Создание простой модели
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch_size', 20])
        
        # Веса
        weights = np.random.randn(20, 1).astype(np.float32)
        bias = np.random.randn(1).astype(np.float32)
        
        W = helper.make_tensor(
            'W',
            TensorProto.FLOAT,
            [20, 1],
            weights.flatten()
        )
        
        B = helper.make_tensor(
            'B',
            TensorProto.FLOAT,
            [1],
            bias.flatten()
        )
        
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input', 'W'],
            outputs=['matmul_output']
        )
        
        add_node = helper.make_node(
            'Add',
            inputs=['matmul_output', 'B'],
            outputs=['add_output']
        )
        
        sigmoid_node = helper.make_node(
            'Sigmoid',
            inputs=['add_output'],
            outputs=['output']
        )
        
        graph = helper.make_graph(
            [matmul_node, add_node, sigmoid_node],
            'test_model',
            [X],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch_size', 1])],
            [W, B]
        )
        
        model = helper.make_model(graph, producer_name='test')
        onnx.save(model, str(self.model_path))
    
    def test_health_endpoint(self):
        """Тест health endpoint"""
        response = self.client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_model_info_endpoint(self):
        """Тест endpoint информации о модели"""
        response = self.client.get("/api/v1/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'model_name' in data
        assert 'version' in data
        assert 'input_shape' in data
        assert 'output_shape' in data
        assert 'framework' in data
    
    def test_single_prediction_endpoint(self):
        """Тест endpoint предсказания для одного образца"""
        # Тестовые данные
        test_data = {
            "age": 35,
            "income": 50000,
            "credit_amount": 10000,
            "loan_duration": 12,
            "payment_to_income": 0.2,
            "existing_credits": 2,
            "dependents": 1,
            "residence_since": 5,
            "installment_rate": 2.0,
            "credit_history": "A30",
            "purpose": "A40",
            "savings": "A61",
            "employment_duration": "A71",
            "personal_status": "A91",
            "debtors": "A101",
            "property": "A121",
            "other_installment_plans": "A141",
            "housing": "A151",
            "job": "A171",
            "telephone": "A191",
            "foreign_worker": "A201"
        }
        
        response = self.client.post(
            "/api/v1/predict",
            json=test_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'prediction' in data
        assert 'probability' in data
        assert 'risk_category' in data
        assert 'request_id' in data
        assert 'timestamp' in data
        
        # Проверка типов данных
        assert isinstance(data['prediction'], int)
        assert isinstance(data['probability'], float)
        assert data['probability'] >= 0 and data['probability'] <= 1
        assert data['risk_category'] in ['low', 'medium', 'high']
    
    def test_batch_prediction_endpoint(self):
        """Тест endpoint предсказания для батча"""
        # Тестовые данные для батча
        test_batch = [
            {
                "age": 35,
                "income": 50000,
                "credit_amount": 10000,
                "loan_duration": 12,
                "payment_to_income": 0.2,
                "existing_credits": 2,
                "dependents": 1,
                "residence_since": 5,
                "installment_rate": 2.0,
                "credit_history": "A30",
                "purpose": "A40",
                "savings": "A61",
                "employment_duration": "A71",
                "personal_status": "A91",
                "debtors": "A101",
                "property": "A121",
                "other_installment_plans": "A141",
                "housing": "A151",
                "job": "A171",
                "telephone": "A191",
                "foreign_worker": "A201"
            },
            {
                "age": 45,
                "income": 75000,
                "credit_amount": 20000,
                "loan_duration": 24,
                "payment_to_income": 0.27,
                "existing_credits": 3,
                "dependents": 2,
                "residence_since": 10,
                "installment_rate": 2.5,
                "credit_history": "A31",
                "purpose": "A41",
                "savings": "A62",
                "employment_duration": "A72",
                "personal_status": "A92",
                "debtors": "A102",
                "property": "A122",
                "other_installment_plans": "A142",
                "housing": "A152",
                "job": "A172",
                "telephone": "A192",
                "foreign_worker": "A201"
            }
        ]
        
        response = self.client.post(
            "/api/v1/predict/batch",
            json={"samples": test_batch}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'predictions' in data
        assert 'probabilities' in data
        assert 'request_id' in data
        assert 'batch_size' in data
        
        assert len(data['predictions']) == 2
        assert len(data['probabilities']) == 2
        
        # Проверка типов данных
        for prediction in data['predictions']:
            assert isinstance(prediction, int)
        
        for probability in data['probabilities']:
            assert isinstance(probability, float)
            assert probability >= 0 and probability <= 1
    
    def test_invalid_data_validation(self):
        """Тест валидации некорректных данных"""
        # Данные с пропущенными полями
        invalid_data = {
            "age": 35,
            "income": 50000
            # Остальные поля отсутствуют
        }
        
        response = self.client.post(
            "/api/v1/predict",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert 'detail' in data
    
    def test_rate_limiting(self):
        """Тест ограничения скорости запросов"""
        test_data = {
            "age": 35,
            "income": 50000,
            "credit_amount": 10000,
            "loan_duration": 12,
            "payment_to_income": 0.2,
            "existing_credits": 2,
            "dependents": 1,
            "residence_since": 5,
            "installment_rate": 2.0,
            "credit_history": "A30",
            "purpose": "A40",
            "savings": "A61",
            "employment_duration": "A71",
            "personal_status": "A91",
            "debtors": "A101",
            "property": "A121",
            "other_installment_plans": "A141",
            "housing": "A151",
            "job": "A171",
            "telephone": "A191",
            "foreign_worker": "A201"
        }
        
        # Многократные запросы для проверки rate limiting
        responses = []
        for _ in range(15):  # Больше лимита по умолчанию
            response = self.client.post("/api/v1/predict", json=test_data)
            responses.append(response.status_code)
        
        # Проверка, что некоторые запросы получили 429
        assert 429 in responses
    
    def test_authentication(self):
        """Тест аутентификации"""
        # Запрос без API ключа
        response = self.client.get("/api/v1/model/info")
        
        # В тестовом режиме аутентификация может быть отключена
        if response.status_code == 401:
            # Тест с неправильным API ключом
            response = self.client.get(
                "/api/v1/model/info",
                headers={"X-API-Key": "invalid-key"}
            )
            assert response.status_code == 401
            
            # Тест с правильным API ключом (если настроен)
            # response = self.client.get(
            #     "/api/v1/model/info",
            #     headers={"X-API-Key": "test-api-key"}
            # )
            # assert response.status_code == 200
    
    def test_metrics_endpoint(self):
        """Тест endpoint метрик"""
        response = self.client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        
        # Проверка, что ответ содержит метрики Prometheus
        content = response.text
        assert 'http_requests_total' in content
        assert 'inference_latency_seconds' in content
    
    def test_detailed_prediction_endpoint(self):
        """Тест endpoint детализированного предсказания"""
        test_data = {
            "age": 35,
            "income": 50000,
            "credit_amount": 10000,
            "loan_duration": 12,
            "payment_to_income": 0.2,
            "existing_credits": 2,
            "dependents": 1,
            "residence_since": 5,
            "installment_rate": 2.0,
            "credit_history": "A30",
            "purpose": "A40",
            "savings": "A61",
            "employment_duration": "A71",
            "personal_status": "A91",
            "debtors": "A101",
            "property": "A121",
            "other_installment_plans": "A141",
            "housing": "A151",
            "job": "A171",
            "telephone": "A191",
            "foreign_worker": "A201"
        }
        
        response = self.client.post(
            "/api/v1/predict/detailed",
            json=test_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверка наличия дополнительных полей
        assert 'prediction' in data
        assert 'probability' in data
        assert 'risk_category' in data
        assert 'explanation' in data
        assert 'feature_importance' in data['explanation']
        assert 'top_features' in data['explanation']
        assert 'confidence_score' in data
    
    def test_version_endpoint(self):
        """Тест endpoint версии API"""
        response = self.client.get("/api/v1/version")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'api_version' in data
        assert 'model_version' in data
        assert 'build_date' in data
    
    def test_error_handling(self):
        """Тест обработки ошибок"""
        # Неверный HTTP метод
        response = self.client.put("/api/v1/predict", json={})
        assert response.status_code == 405
        
        # Неверный content-type
        response = self.client.post(
            "/api/v1/predict",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415
        
        # Слишком большой запрос
        large_data = {"data": "x" * (10 * 1024 * 1024)}  # 10MB
        response = self.client.post("/api/v1/predict", json=large_data)
        assert response.status_code == 413
    
    @pytest.mark.parametrize("endpoint", [
        "/api/v1/predict",
        "/api/v1/predict/batch",
        "/api/v1/model/info",
        "/api/v1/health"
    ])
    def test_cors_headers(self, endpoint):
        """Тест CORS headers для разных endpoints"""
        response = self.client.options(
            endpoint,
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # Проверка CORS headers
        if response.status_code == 200:
            assert "access-control-allow-origin" in response.headers
            assert "access-control-allow-methods" in response.headers
    
    def test_performance_monitoring(self):
        """Тест мониторинга производительности"""
        import time
        
        test_data = {
            "age": 35,
            "income": 50000,
            "credit_amount": 10000,
            "loan_duration": 12,
            "payment_to_income": 0.2,
            "existing_credits": 2,
            "dependents": 1,
            "residence_since": 5,
            "installment_rate": 2.0,
            "credit_history": "A30",
            "purpose": "A40",
            "savings": "A61",
            "employment_duration": "A71",
            "personal_status": "A91",
            "debtors": "A101",
            "property": "A121",
            "other_installment_plans": "A141",
            "housing": "A151",
            "job": "A171",
            "telephone": "A191",
            "foreign_worker": "A201"
        }
        
        # Измерение времени выполнения
        start_time = time.time()
        
        for _ in range(10):
            response = self.client.post("/api/v1/predict", json=test_data)
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nAverage API response time: {total_time/10*1000:.2f}ms")
        
        # Проверка метрик производительности
        metrics_response = self.client.get("/api/v1/metrics")
        assert metrics_response.status_code == 200
        
        metrics_text = metrics_response.text
        assert 'inference_latency_seconds_count' in metrics_text
    
    def test_concurrent_requests(self):
        """Тест конкурентных запросов"""
        import threading
        import queue
        
        test_data = {
            "age": 35,
            "income": 50000,
            "credit_amount": 10000,
            "loan_duration": 12,
            "payment_to_income": 0.2,
            "existing_credits": 2,
            "dependents": 1,
            "residence_since": 5,
            "installment_rate": 2.0,
            "credit_history": "A30",
            "purpose": "A40",
            "savings": "A61",
            "employment_duration": "A71",
            "personal_status": "A91",
            "debtors": "A101",
            "property": "A121",
            "other_installment_plans": "A141",
            "housing": "A151",
            "job": "A171",
            "telephone": "A191",
            "foreign_worker": "A201"
        }
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = self.client.post("/api/v1/predict", json=test_data)
                results.put(response.status_code)
            except Exception as e:
                results.put(str(e))
        
        # Запуск конкурентных запросов
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)
        
        # Ожидание завершения всех потоков
        for thread in threads:
            thread.join()
        
        # Проверка результатов
        status_codes = []
        while not results.empty():
            result = results.get()
            if isinstance(result, int):
                status_codes.append(result)
        
        # Все запросы должны завершиться успешно
        assert all(code == 200 for code in status_codes)

class TestModelManagementEndpoints:
    """Тесты для endpoints управления моделью"""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Настройка тестового окружения"""
        self.test_dir = tmp_path
        self.models_dir = self.test_dir / "models"
        self.models_dir.mkdir()
        
        # Создание нескольких версий моделей
        for version in ["v1.0.0", "v1.1.0", "v2.0.0"]:
            version_dir = self.models_dir / version
            version_dir.mkdir()
            
            # Создание простой модели
            self._create_simple_model(version_dir / "model.onnx")
        
        # Создание тестового клиента
        self.client = TestClient(app)
        
        # Переопределение конфигурации
        app.state.config = {
            'model': {
                'path': str(self.models_dir),
                'current_version': 'v1.1.0',
                'input_shape': (1, 20)
            }
        }
        
        yield
        
        # Очистка
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_simple_model(self, model_path):
        """Создание простой модели"""
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
        
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 20])
        
        weights = np.ones((20, 1), dtype=np.float32) * 0.05
        bias = np.array([0.0], dtype=np.float32)
        
        W = helper.make_tensor(
            'W',
            TensorProto.FLOAT,
            [20, 1],
            weights.flatten()
        )
        
        B = helper.make_tensor(
            'B',
            TensorProto.FLOAT,
            [1],
            bias.flatten()
        )
        
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input', 'W'],
            outputs=['matmul_output']
        )
        
        add_node = helper.make_node(
            'Add',
            inputs=['matmul_output', 'B'],
            outputs=['add_output']
        )
        
        sigmoid_node = helper.make_node(
            'Sigmoid',
            inputs=['add_output'],
            outputs=['output']
        )
        
        graph = helper.make_graph(
            [matmul_node, add_node, sigmoid_node],
            'simple_model',
            [X],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])],
            [W, B]
        )
        
        model = helper.make_model(graph, producer_name='test')
        onnx.save(model, str(model_path))
    
    def test_list_models_endpoint(self):
        """Тест endpoint списка моделей"""
        response = self.client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'models' in data
        assert 'current_model' in data
        
        models = data['models']
        assert len(models) == 3
        
        # Проверка информации о каждой модели
        for model in models:
            assert 'version' in model
            assert 'path' in model
            assert 'created_at' in model
            assert 'size_mb' in model
    
    def test_switch_model_endpoint(self):
        """Тест endpoint переключения модели"""
        # Переключение на другую версию
        switch_data = {
            "version": "v2.0.0"
        }
        
        response = self.client.post(
            "/api/v1/models/switch",
            json=switch_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'success' in data
        assert data['success'] == True
        assert 'previous_version' in data
        assert 'new_version' in data
        assert data['new_version'] == "v2.0.0"
    
    def test_switch_to_invalid_model(self):
        """Тест переключения на несуществующую модель"""
        switch_data = {
            "version": "v3.0.0"  # Не существует
        }
        
        response = self.client.post(
            "/api/v1/models/switch",
            json=switch_data
        )
        
        assert response.status_code == 404
        data = response.json()
        assert 'detail' in data
    
    def test_model_comparison_endpoint(self):
        """Тест endpoint сравнения моделей"""
        compare_data = {
            "model_a": "v1.0.0",
            "model_b": "v2.0.0",
            "test_data_size": 10
        }
        
        response = self.client.post(
            "/api/v1/models/compare",
            json=compare_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'comparison_results' in data
        assert 'model_a' in data['comparison_results']
        assert 'model_b' in data['comparison_results']
        assert 'metrics' in data['comparison_results']
        
        metrics = data['comparison_results']['metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_model_performance_endpoint(self):
        """Тест endpoint производительности модели"""
        response = self.client.get("/api/v1/models/v1.1.0/performance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'model_version' in data
        assert 'performance_metrics' in data
        assert 'inference_stats' in data
        
        metrics = data['performance_metrics']
        assert 'total_predictions' in metrics
        assert 'average_latency_ms' in metrics
        assert 'success_rate' in metrics
    
    def test_model_retrain_endpoint(self):
        """Тест endpoint переобучения модели"""
        retrain_data = {
            "trigger": "manual",
            "description": "Manual retraining test",
            "parameters": {
                "epochs": 10,
                "learning_rate": 0.001
            }
        }
        
        response = self.client.post(
            "/api/v1/models/retrain",
            json=retrain_data
        )
        
        # Проверяем, что endpoint существует
        # В реальной системе это запускало бы асинхронную задачу
        assert response.status_code in [200, 202]
        
        if response.status_code == 200:
            data = response.json()
            assert 'training_id' in data
            assert 'status' in data
            assert 'estimated_completion' in data
    
    def test_model_backup_endpoint(self):
        """Тест endpoint backup модели"""
        response = self.client.post("/api/v1/models/v1.1.0/backup")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'backup_id' in data
        assert 'backup_path' in data
        assert 'model_version' in data
        assert 'timestamp' in data
    
    def test_model_rollback_endpoint(self):
        """Тест endpoint отката модели"""
        # Сначала создаем backup
        backup_response = self.client.post("/api/v1/models/v1.1.0/backup")
        backup_data = backup_response.json()
        backup_id = backup_data['backup_id']
        
        # Пробуем откатиться
        rollback_data = {
            "backup_id": backup_id
        }
        
        response = self.client.post(
            "/api/v1/models/rollback",
            json=rollback_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'success' in data
        assert 'restored_version' in data
        assert 'backup_id' in data
    
    def test_model_health_check_endpoint(self):
        """Тест endpoint проверки здоровья модели"""
        response = self.client.get("/api/v1/models/v1.1.0/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'model_version' in data
        assert 'status' in data
        assert 'checks' in data
        
        checks = data['checks']
        assert 'model_loaded' in checks
        assert 'model_valid' in checks
        assert 'performance_check' in checks
        
        # Все проверки должны пройти
        for check_name, check_result in checks.items():
            assert 'passed' in check_result
            assert check_result['passed'] == True

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])