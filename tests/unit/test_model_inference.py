"""
Модульные тесты для инференса модели
"""
import unittest
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from pathlib import Path
import json
import tempfile
import sys
import os

# Добавление пути к исходному коду
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml_pipeline.inference.model_loader import ModelLoader
from src.ml_pipeline.inference.predictor import Predictor
from src.utils.data_processing import DataProcessor

class TestModelLoader(unittest.TestCase):
    """Тесты для загрузчика моделей"""
    
    def setUp(self):
        """Подготовка тестового окружения"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = Path(self.test_dir) / "test_model.onnx"
        
        # Создание простой ONNX модели для тестов
        self._create_test_onnx_model()
        
        self.config = {
            'model': {
                'path': str(self.model_path),
                'input_shape': (1, 20),
                'output_shape': (1, 1),
                'framework': 'onnx'
            }
        }
    
    def _create_test_onnx_model(self):
        """Создание тестовой ONNX модели"""
        import onnx
        from onnx import helper, TensorProto
        
        # Создание простого графа
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 20])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
        
        node = helper.make_node(
            'MatMul',
            inputs=['X', 'W'],
            outputs=['Y']
        )
        
        # Создание весов
        W = helper.make_tensor(
            'W',
            TensorProto.FLOAT,
            [20, 1],
            np.random.randn(20, 1).astype(np.float32).flatten()
        )
        
        graph = helper.make_graph(
            [node],
            'test_model',
            [X],
            [Y],
            [W]
        )
        
        model = helper.make_model(graph, producer_name='test')
        
        # Сохранение модели
        onnx.save(model, str(self.model_path))
    
    def tearDown(self):
        """Очистка после тестов"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_model_loader_initialization(self):
        """Тест инициализации загрузчика моделей"""
        loader = ModelLoader(self.config)
        
        self.assertIsNotNone(loader)
        self.assertEqual(loader.framework, 'onnx')
        self.assertTrue(loader.model_path.exists())
    
    def test_load_onnx_model(self):
        """Тест загрузки ONNX модели"""
        loader = ModelLoader(self.config)
        session = loader.load_model()
        
        self.assertIsNotNone(session)
        self.assertIsInstance(session, ort.InferenceSession)
    
    def test_model_loader_with_invalid_path(self):
        """Тест с некорректным путем к модели"""
        invalid_config = self.config.copy()
        invalid_config['model']['path'] = "/invalid/path/model.onnx"
        
        with self.assertRaises(FileNotFoundError):
            ModelLoader(invalid_config)
    
    def test_warmup_model(self):
        """Тест прогрева модели"""
        loader = ModelLoader(self.config)
        loader.load_model()
        
        # Прогрев модели
        warmup_time = loader.warmup(num_iterations=10)
        
        self.assertIsInstance(warmup_time, float)
        self.assertGreater(warmup_time, 0)
    
    def test_model_metadata(self):
        """Тест получения метаданных модели"""
        loader = ModelLoader(self.config)
        loader.load_model()
        
        metadata = loader.get_model_metadata()
        
        self.assertIn('inputs', metadata)
        self.assertIn('outputs', metadata)
        self.assertIn('providers', metadata)
        
        # Проверка входов
        inputs = metadata['inputs']
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0]['name'], 'X')
        self.assertEqual(inputs[0]['shape'], [1, 20])
    
    def test_model_versioning(self):
        """Тест версионирования моделей"""
        loader = ModelLoader(self.config)
        
        # Создание нескольких версий модели
        model_dir = Path(self.test_dir) / "models"
        model_dir.mkdir(exist_ok=True)
        
        versions = ['v1.0.0', 'v1.1.0', 'v2.0.0']
        for version in versions:
            version_dir = model_dir / version
            version_dir.mkdir(exist_ok=True)
            
            # Копирование модели в каждую версию
            import shutil
            shutil.copy(self.model_path, version_dir / "model.onnx")
        
        # Тест выбора версии
        config_with_versions = {
            'model': {
                'path': str(model_dir),
                'version': 'v1.1.0',
                'input_shape': (1, 20)
            }
        }
        
        loader = ModelLoader(config_with_versions)
        session = loader.load_model()
        
        self.assertIsNotNone(session)

class TestPredictor(unittest.TestCase):
    """Тесты для предиктора"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = Path(self.test_dir) / "test_model.onnx"
        
        # Создание ONNX модели
        self._create_test_onnx_model()
        
        self.config = {
            'model': {
                'path': str(self.model_path),
                'input_shape': (1, 20),
                'output_shape': (1, 1),
                'framework': 'onnx'
            },
            'preprocessing': {
                'scaler_path': None,
                'categorical_encoding': 'onehot'
            }
        }
        
        # Тестовые данные
        self.sample_data = pd.DataFrame({
            'age': [35],
            'income': [50000],
            'credit_amount': [10000],
            'loan_duration': [12],
            'payment_to_income': [0.2],
            'existing_credits': [2],
            'dependents': [1],
            'residence_since': [5],
            'installment_rate': [2.0],
            'credit_history': ['A30'],
            'purpose': ['A40'],
            'savings': ['A61'],
            'employment_duration': ['A71'],
            'personal_status': ['A91'],
            'debtors': ['A101'],
            'property': ['A121'],
            'other_installment_plans': ['A141'],
            'housing': ['A151'],
            'job': ['A171'],
            'telephone': ['A191'],
            'foreign_worker': ['A201']
        })
        
        self.batch_data = pd.DataFrame({
            'age': [35, 40, 25],
            'income': [50000, 60000, 30000],
            'credit_amount': [10000, 15000, 5000],
            'loan_duration': [12, 24, 6],
            'payment_to_income': [0.2, 0.25, 0.15],
            'existing_credits': [2, 3, 1],
            'dependents': [1, 2, 0],
            'residence_since': [5, 10, 2],
            'installment_rate': [2.0, 2.5, 1.5],
            'credit_history': ['A30', 'A31', 'A30'],
            'purpose': ['A40', 'A41', 'A40'],
            'savings': ['A61', 'A62', 'A61'],
            'employment_duration': ['A71', 'A72', 'A71'],
            'personal_status': ['A91', 'A92', 'A91'],
            'debtors': ['A101', 'A102', 'A101'],
            'property': ['A121', 'A122', 'A121'],
            'other_installment_plans': ['A141', 'A142', 'A141'],
            'housing': ['A151', 'A152', 'A151'],
            'job': ['A171', 'A172', 'A171'],
            'telephone': ['A191', 'A192', 'A191'],
            'foreign_worker': ['A201', 'A201', 'A201']
        })
    
    def _create_test_onnx_model(self):
        """Создание тестовой ONNX модели с логистической регрессией"""
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
        
        # Входные данные
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch_size', 45])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch_size', 1])
        
        # Веса и смещение
        weights = np.random.randn(45, 1).astype(np.float32)
        bias = np.random.randn(1).astype(np.float32)
        
        W = helper.make_tensor(
            'W',
            TensorProto.FLOAT,
            [45, 1],
            weights.flatten()
        )
        
        B = helper.make_tensor(
            'B',
            TensorProto.FLOAT,
            [1],
            bias.flatten()
        )
        
        # Узлы графа
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
        
        # Граф
        graph = helper.make_graph(
            [matmul_node, add_node, sigmoid_node],
            'test_model',
            [X],
            [Y],
            [W, B]
        )
        
        # Модель
        model = helper.make_model(
            graph,
            producer_name='test',
            opset_imports=[helper.make_opsetid("", 11)]
        )
        
        # Сохранение
        onnx.save(model, str(self.model_path))
    
    def tearDown(self):
        """Очистка после тестов"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_predictor_initialization(self):
        """Тест инициализации предиктора"""
        predictor = Predictor(self.config)
        
        self.assertIsNotNone(predictor)
        self.assertIsNotNone(predictor.model_loader)
        self.assertIsNotNone(predictor.data_processor)
    
    def test_single_prediction(self):
        """Тест предсказания для одного образца"""
        predictor = Predictor(self.config)
        
        # Предсказание
        result = predictor.predict(self.sample_data)
        
        # Проверка результата
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        self.assertIn('probabilities', result)
        self.assertIn('metadata', result)
        
        predictions = result['predictions']
        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0], [0, 1])
        
        probabilities = result['probabilities']
        self.assertEqual(len(probabilities), 1)
        self.assertGreaterEqual(probabilities[0], 0)
        self.assertLessEqual(probabilities[0], 1)
    
    def test_batch_prediction(self):
        """Тест предсказания для батча"""
        predictor = Predictor(self.config)
        
        # Предсказание для батча
        result = predictor.predict_batch(self.batch_data)
        
        # Проверка результата
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        self.assertIn('probabilities', result)
        
        predictions = result['predictions']
        self.assertEqual(len(predictions), len(self.batch_data))
        
        probabilities = result['probabilities']
        self.assertEqual(len(probabilities), len(self.batch_data))
    
    def test_prediction_with_explanation(self):
        """Тест предсказания с объяснением"""
        config_with_shap = self.config.copy()
        config_with_shap['explanation'] = {
            'enable': True,
            'method': 'shap',
            'sample_size': 100
        }
        
        predictor = Predictor(config_with_shap)
        
        # Предсказание с объяснением
        result = predictor.predict_with_explanation(self.sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('explanation', result)
        self.assertIn('feature_importance', result['explanation'])
    
    def test_prediction_performance(self):
        """Тест производительности предсказаний"""
        predictor = Predictor(self.config)
        
        # Измерение времени выполнения
        import time
        
        start_time = time.time()
        for _ in range(100):
            predictor.predict(self.sample_data)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        print(f"\nAverage prediction time: {avg_time*1000:.2f}ms")
        
        # Проверка, что среднее время меньше 100ms
        self.assertLess(avg_time, 0.1)  # 100ms
    
    def test_error_handling(self):
        """Тест обработки ошибок"""
        predictor = Predictor(self.config)
        
        # Тест с некорректными данными
        invalid_data = pd.DataFrame({
            'invalid_column': [1, 2, 3]
        })
        
        with self.assertRaises(ValueError):
            predictor.predict(invalid_data)
        
        # Тест с пустыми данными
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            predictor.predict(empty_data)
    
    def test_prediction_consistency(self):
        """Тест консистентности предсказаний"""
        predictor = Predictor(self.config)
        
        # Многократные предсказания для одних данных
        results = []
        for _ in range(10):
            result = predictor.predict(self.sample_data)
            results.append(result['probabilities'][0])
        
        # Проверка, что предсказания одинаковые (с допуском)
        first_prob = results[0]
        for prob in results[1:]:
            self.assertAlmostEqual(prob, first_prob, places=5)
    
    def test_threshold_adjustment(self):
        """Тест настройки порога классификации"""
        config_with_threshold = self.config.copy()
        config_with_threshold['classification'] = {
            'threshold': 0.7
        }
        
        predictor = Predictor(config_with_threshold)
        
        result = predictor.predict(self.sample_data)
        probability = result['probabilities'][0]
        prediction = result['predictions'][0]
        
        # Проверка, что предсказание соответствует порогу
        if probability >= 0.7:
            self.assertEqual(prediction, 1)
        else:
            self.assertEqual(prediction, 0)

class TestDataProcessor(unittest.TestCase):
    """Тесты для обработки данных"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.numerical_features = ['age', 'income', 'credit_amount']
        self.categorical_features = ['credit_history', 'purpose']
        
        self.config = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'preprocessing': {
                'scaling': 'standard',
                'categorical_encoding': 'onehot'
            }
        }
        
        # Тестовые данные
        self.train_data = pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'income': [30000, 40000, 50000, 60000, 70000],
            'credit_amount': [5000, 10000, 15000, 20000, 25000],
            'credit_history': ['A30', 'A31', 'A30', 'A32', 'A31'],
            'purpose': ['A40', 'A41', 'A40', 'A42', 'A41']
        })
        
        self.test_data = pd.DataFrame({
            'age': [30, 50],
            'income': [35000, 55000],
            'credit_amount': [7500, 17500],
            'credit_history': ['A30', 'A31'],
            'purpose': ['A40', 'A42']
        })
    
    def test_data_processor_initialization(self):
        """Тест инициализации процессора данных"""
        processor = DataProcessor(self.config)
        
        self.assertIsNotNone(processor)
        self.assertEqual(processor.numerical_features, self.numerical_features)
        self.assertEqual(processor.categorical_features, self.categorical_features)
    
    def test_fit_transform(self):
        """Тест обучения и трансформации данных"""
        processor = DataProcessor(self.config)
        
        # Обучение и трансформация тренировочных данных
        transformed_train = processor.fit_transform(self.train_data)
        
        self.assertIsInstance(transformed_train, np.ndarray)
        
        # Проверка размерности (5 samples, n_features)
        self.assertEqual(transformed_train.shape[0], len(self.train_data))
        self.assertGreater(transformed_train.shape[1], len(self.numerical_features))
    
    def test_transform_only(self):
        """Тест трансформации без обучения"""
        processor = DataProcessor(self.config)
        
        # Обучение на тренировочных данных
        processor.fit(self.train_data)
        
        # Трансформация тестовых данных
        transformed_test = processor.transform(self.test_data)
        
        self.assertIsInstance(transformed_test, np.ndarray)
        self.assertEqual(transformed_test.shape[0], len(self.test_data))
    
    def test_missing_value_handling(self):
        """Тест обработки пропущенных значений"""
        data_with_nans = self.train_data.copy()
        data_with_nans.loc[0, 'age'] = np.nan
        data_with_nans.loc[1, 'income'] = np.nan
        
        config_with_missing = self.config.copy()
        config_with_missing['preprocessing']['missing_values'] = 'impute'
        
        processor = DataProcessor(config_with_missing)
        
        # Трансформация данных с пропусками
        transformed = processor.fit_transform(data_with_nans)
        
        self.assertIsInstance(transformed, np.ndarray)
        self.assertFalse(np.any(np.isnan(transformed)))
    
    def test_outlier_handling(self):
        """Тест обработки выбросов"""
        config_with_outliers = self.config.copy()
        config_with_outliers['preprocessing']['outlier_detection'] = True
        config_with_outliers['preprocessing']['outlier_method'] = 'zscore'
        config_with_outliers['preprocessing']['outlier_threshold'] = 3
        
        processor = DataProcessor(config_with_outliers)
        
        # Трансформация с обработкой выбросов
        transformed = processor.fit_transform(self.train_data)
        
        self.assertIsInstance(transformed, np.ndarray)
    
    def test_feature_engineering(self):
        """Тест создания новых признаков"""
        config_with_engineering = self.config.copy()
        config_with_engineering['feature_engineering'] = {
            'debt_to_income': 'credit_amount / income',
            'age_group': """
                case
                    when age < 30 then 'young'
                    when age between 30 and 50 then 'middle'
                    else 'senior'
                end
            """
        }
        
        processor = DataProcessor(config_with_engineering)
        
        # Трансформация с созданием признаков
        transformed = processor.fit_transform(self.train_data)
        
        self.assertIsInstance(transformed, np.ndarray)
        # Проверка, что добавлены новые признаки
        self.assertGreater(transformed.shape[1], len(self.numerical_features) + 2)
    
    def test_save_load_processor(self):
        """Тест сохранения и загрузки процессора"""
        processor = DataProcessor(self.config)
        processor.fit(self.train_data)
        
        # Сохранение процессора
        import pickle
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            pickle.dump(processor, f)
            processor_path = f.name
        
        try:
            # Загрузка процессора
            with open(processor_path, 'rb') as f:
                loaded_processor = pickle.load(f)
            
            # Проверка, что загруженный процессор работает
            transformed = loaded_processor.transform(self.test_data)
            self.assertIsInstance(transformed, np.ndarray)
            
        finally:
            # Очистка
            os.unlink(processor_path)
    
    def test_data_validation(self):
        """Тест валидации данных"""
        processor = DataProcessor(self.config)
        
        # Данные с некорректным типом
        invalid_data = pd.DataFrame({
            'age': ['twenty', 'thirty'],  # Строки вместо чисел
            'income': [30000, 40000],
            'credit_amount': [5000, 10000],
            'credit_history': ['A30', 'A31'],
            'purpose': ['A40', 'A41']
        })
        
        with self.assertRaises(ValueError):
            processor.fit(invalid_data)
        
        # Данные с неизвестными категориями
        data_unknown_cat = pd.DataFrame({
            'age': [25, 35],
            'income': [30000, 40000],
            'credit_amount': [5000, 10000],
            'credit_history': ['UNKNOWN', 'A31'],  # Неизвестная категория
            'purpose': ['A40', 'A41']
        })
        
        # Это должно работать, так как категории будут обработаны
        processor.fit(self.train_data)
        transformed = processor.transform(data_unknown_cat)
        
        self.assertIsInstance(transformed, np.ndarray)

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def setUp(self):
        """Подготовка интеграционного теста"""
        self.test_dir = tempfile.mkdtemp()
        
        # Создание полной конфигурации
        self.config = {
            'model': {
                'path': str(Path(self.test_dir) / 'model.onnx'),
                'input_shape': (1, 45),
                'output_shape': (1, 1),
                'framework': 'onnx'
            },
            'data': {
                'numerical_features': ['age', 'income', 'credit_amount'],
                'categorical_features': ['credit_history', 'purpose'],
                'preprocessing': {
                    'scaling': 'standard',
                    'categorical_encoding': 'onehot'
                }
            }
        }
        
        # Создание тестовой модели
        self._create_integration_model()
    
    def _create_integration_model(self):
        """Создание модели для интеграционного теста"""
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
        
        # Простая модель для интеграционного теста
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch_size', 7])  # 3 numerical + 4 onehot
        
        # Веса для 7 признаков
        weights = np.ones((7, 1), dtype=np.float32) * 0.1
        bias = np.array([0.0], dtype=np.float32)
        
        W = helper.make_tensor(
            'W',
            TensorProto.FLOAT,
            [7, 1],
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
            'integration_model',
            [X],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch_size', 1])],
            [W, B]
        )
        
        model = helper.make_model(graph, producer_name='integration_test')
        onnx.save(model, self.config['model']['path'])
    
    def tearDown(self):
        """Очистка после тестов"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_end_to_end_prediction(self):
        """End-to-end тест предсказания"""
        from src.ml_pipeline.inference.predictor import Predictor
        
        predictor = Predictor(self.config)
        
        # Тестовые данные
        test_data = pd.DataFrame({
            'age': [30, 40],
            'income': [35000, 45000],
            'credit_amount': [8000, 12000],
            'credit_history': ['A30', 'A31'],
            'purpose': ['A40', 'A41']
        })
        
        # Предсказание
        result = predictor.predict(test_data)
        
        # Проверка результата
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        self.assertIn('probabilities', result)
        
        predictions = result['predictions']
        self.assertEqual(len(predictions), 2)
        
        probabilities = result['probabilities']
        self.assertEqual(len(probabilities), 2)
        
        # Проверка, что вероятности в диапазоне [0, 1]
        for prob in probabilities:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
    
    def test_batch_end_to_end(self):
        """End-to-end тест для батча"""
        from src.ml_pipeline.inference.predictor import Predictor
        
        predictor = Predictor(self.config)
        
        # Большой батч данных
        batch_size = 100
        test_data = pd.DataFrame({
            'age': np.random.randint(20, 60, batch_size),
            'income': np.random.randint(20000, 80000, batch_size),
            'credit_amount': np.random.randint(1000, 30000, batch_size),
            'credit_history': np.random.choice(['A30', 'A31', 'A32'], batch_size),
            'purpose': np.random.choice(['A40', 'A41', 'A42'], batch_size)
        })
        
        # Предсказание для батча
        result = predictor.predict_batch(test_data)
        
        # Проверка результата
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result['predictions']), batch_size)
        self.assertEqual(len(result['probabilities']), batch_size)
        
        # Проверка производительности
        import time
        start_time = time.time()
        predictor.predict_batch(test_data)
        execution_time = time.time() - start_time
        
        print(f"\nBatch prediction time for {batch_size} samples: {execution_time:.3f}s")
        self.assertLess(execution_time, 1.0)  # Должно быть меньше 1 секунды

if __name__ == "__main__":
    # Запуск всех тестов
    unittest.main(verbosity=2)