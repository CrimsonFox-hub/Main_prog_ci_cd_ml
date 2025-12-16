"""
Нагрузочное тестирование API с использованием Locust
"""
from locust import HttpUser, task, between, constant, LoadTestShape
import json
import random
from datetime import datetime
import logging

class CreditScoringUser(HttpUser):
    """Пользователь для нагрузочного тестирования кредитного скоринга"""
    
    # Время между задачами: 1-3 секунды
    wait_time = between(1, 3)
    
    # Общие заголовки
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "load-test-key"
    }
    
    def on_start(self):
        """Действия при старте пользователя"""
        logging.info(f"User {self.user_id} started")
        
        # Генератор случайных данных
        self.data_generator = DataGenerator()
        
        # Сбор метрик
        self.metrics = {
            'requests': 0,
            'errors': 0,
            'total_latency': 0
        }
    
    def on_stop(self):
        """Действия при остановке пользователя"""
        if self.metrics['requests'] > 0:
            avg_latency = self.metrics['total_latency'] / self.metrics['requests']
            error_rate = (self.metrics['errors'] / self.metrics['requests']) * 100
            
            logging.info(
                f"User {self.user_id} finished: "
                f"{self.metrics['requests']} requests, "
                f"{self.metrics['errors']} errors ({error_rate:.1f}%), "
                f"avg latency: {avg_latency:.2f}ms"
            )
    
    @task(5)  # Частота выполнения: 5
    def test_single_prediction(self):
        """Тест предсказания для одного заемщика"""
        data = self.data_generator.generate_single_application()
        
        with self.client.post(
            "/api/v1/predict",
            json=data,
            headers=self.headers,
            name="Single Prediction",
            catch_response=True
        ) as response:
            self._process_response(response, "single_prediction")
    
    @task(3)  # Частота выполнения: 3
    def test_batch_prediction(self):
        """Тест предсказания для батча заемщиков"""
        batch_size = random.randint(2, 10)
        data = {
            "samples": self.data_generator.generate_batch_applications(batch_size)
        }
        
        with self.client.post(
            "/api/v1/predict/batch",
            json=data,
            headers=self.headers,
            name=f"Batch Prediction ({batch_size})",
            catch_response=True
        ) as response:
            self._process_response(response, "batch_prediction")
    
    @task(1)  # Частота выполнения: 1
    def test_health_check(self):
        """Тест health check endpoint"""
        with self.client.get(
            "/api/v1/health",
            headers=self.headers,
            name="Health Check",
            catch_response=True
        ) as response:
            self._process_response(response, "health_check")
    
    @task(1)  # Частота выполнения: 1
    def test_model_info(self):
        """Тест получения информации о модели"""
        with self.client.get(
            "/api/v1/model/info",
            headers=self.headers,
            name="Model Info",
            catch_response=True
        ) as response:
            self._process_response(response, "model_info")
    
    @task(2)  # Частота выполнения: 2
    def test_detailed_prediction(self):
        """Тест детализированного предсказания"""
        data = self.data_generator.generate_single_application()
        
        with self.client.post(
            "/api/v1/predict/detailed",
            json=data,
            headers=self.headers,
            name="Detailed Prediction",
            catch_response=True
        ) as response:
            self._process_response(response, "detailed_prediction")
    
    def _process_response(self, response, endpoint_name):
        """Обработка ответа и сбор метрик"""
        self.metrics['requests'] += 1
        self.metrics['total_latency'] += response.elapsed.total_seconds() * 1000
        
        # Проверка статуса ответа
        if response.status_code != 200:
            self.metrics['errors'] += 1
            response.failure(f"Status {response.status_code}: {response.text}")
            
            # Логирование ошибок
            if response.status_code >= 500:
                logging.error(
                    f"Server error on {endpoint_name}: "
                    f"{response.status_code} - {response.text}"
                )
        else:
            # Проверка содержимого ответа
            try:
                data = response.json()
                
                # Дополнительные проверки для разных endpoints
                if endpoint_name == "single_prediction":
                    if "prediction" not in data or "probability" not in data:
                        response.failure("Invalid response format")
                        self.metrics['errors'] += 1
                    
                    # Проверка допустимости вероятности
                    probability = data.get("probability", -1)
                    if probability < 0 or probability > 1:
                        response.failure(f"Invalid probability: {probability}")
                        self.metrics['errors'] += 1
                
                elif endpoint_name == "batch_prediction":
                    predictions = data.get("predictions", [])
                    probabilities = data.get("probabilities", [])
                    
                    if len(predictions) == 0 or len(probabilities) == 0:
                        response.failure("Empty batch response")
                        self.metrics['errors'] += 1
                
                elif endpoint_name == "health_check":
                    if data.get("status") != "healthy":
                        response.failure("Health check failed")
                        self.metrics['errors'] += 1
                
                response.success()
                
            except json.JSONDecodeError:
                response.failure("Invalid JSON response")
                self.metrics['errors'] += 1
                logging.error(f"JSON decode error on {endpoint_name}: {response.text}")

class DataGenerator:
    """Генератор тестовых данных для кредитных заявок"""
    
    # Категориальные признаки
    CREDIT_HISTORY = ["A30", "A31", "A32", "A33", "A34"]
    PURPOSE = ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"]
    SAVINGS = ["A61", "A62", "A63", "A64", "A65"]
    EMPLOYMENT_DURATION = ["A71", "A72", "A73", "A74", "A75"]
    PERSONAL_STATUS = ["A91", "A92", "A93", "A94"]
    DEBTORS = ["A101", "A102", "A103"]
    PROPERTY = ["A121", "A122", "A123", "A124"]
    OTHER_INSTALLMENT_PLANS = ["A141", "A142", "A143"]
    HOUSING = ["A151", "A152", "A153"]
    JOB = ["A171", "A172", "A173", "A174"]
    TELEPHONE = ["A191", "A192"]
    FOREIGN_WORKER = ["A201", "A202"]
    
    def generate_single_application(self):
        """Генерация данных для одной заявки"""
        return {
            "age": random.randint(18, 70),
            "income": random.randint(20000, 150000),
            "credit_amount": random.randint(1000, 50000),
            "loan_duration": random.randint(6, 60),
            "payment_to_income": round(random.uniform(0.1, 0.5), 2),
            "existing_credits": random.randint(0, 5),
            "dependents": random.randint(0, 5),
            "residence_since": random.randint(0, 20),
            "installment_rate": round(random.uniform(1.0, 4.0), 1),
            "credit_history": random.choice(self.CREDIT_HISTORY),
            "purpose": random.choice(self.PURPOSE),
            "savings": random.choice(self.SAVINGS),
            "employment_duration": random.choice(self.EMPLOYMENT_DURATION),
            "personal_status": random.choice(self.PERSONAL_STATUS),
            "debtors": random.choice(self.DEBTORS),
            "property": random.choice(self.PROPERTY),
            "other_installment_plans": random.choice(self.OTHER_INSTALLMENT_PLANS),
            "housing": random.choice(self.HOUSING),
            "job": random.choice(self.JOB),
            "telephone": random.choice(self.TELEPHONE),
            "foreign_worker": random.choice(self.FOREIGN_WORKER)
        }
    
    def generate_batch_applications(self, batch_size):
        """Генерация данных для батча заявок"""
        return [self.generate_single_application() for _ in range(batch_size)]
    
    def generate_edge_cases(self):
        """Генерация крайних случаев для тестирования"""
        edge_cases = []
        
        # 1. Молодой заемщик с низким доходом
        edge_cases.append({
            "age": 18,
            "income": 10000,
            "credit_amount": 50000,
            "loan_duration": 60,
            "payment_to_income": 0.5,
            "existing_credits": 5,
            "dependents": 5,
            "residence_since": 0,
            "installment_rate": 4.0,
            "credit_history": "A34",  # Плохая кредитная история
            "purpose": "A40",  # Новая машина
            "savings": "A65",  # Нет сбережений
            "employment_duration": "A71",  # Безработный
            "personal_status": "A94",  # Разведен/разведена
            "debtors": "A101",  # Есть поручители
            "property": "A121",  # Недвижимость
            "other_installment_plans": "A143",  # Есть другие планы
            "housing": "A151",  # Съемное жилье
            "job": "A171",  # Неквалифицированный
            "telephone": "A191",  # Нет телефона
            "foreign_worker": "A202"  # Иностранец
        })
        
        # 2. Зрелый заемщик с высоким доходом
        edge_cases.append({
            "age": 65,
            "income": 200000,
            "credit_amount": 10000,
            "loan_duration": 12,
            "payment_to_income": 0.05,
            "existing_credits": 0,
            "dependents": 0,
            "residence_since": 20,
            "installment_rate": 1.0,
            "credit_history": "A30",  # Отличная кредитная история
            "purpose": "A410",  # Бизнес
            "savings": "A61",  # Большие сбережения
            "employment_duration": "A75",  > 7 лет
            "personal_status": "A91",  # Мужчина, разведен
            "debtors": "A101",  # Нет поручителей
            "property": "A124",  # Недвижимость + страхование жизни
            "other_installment_plans": "A141",  # Банк
            "housing": "A153",  # Собственное жилье
            "job": "A174",  # Руководитель
            "telephone": "A192",  # Есть телефон
            "foreign_worker": "A201"  # Гражданин
        })
        
        # 3. Средний случай
        edge_cases.append({
            "age": 35,
            "income": 50000,
            "credit_amount": 20000,
            "loan_duration": 36,
            "payment_to_income": 0.4,
            "existing_credits": 2,
            "dependents": 2,
            "residence_since": 5,
            "installment_rate": 2.5,
            "credit_history": "A31",  # Хорошая кредитная история
            "purpose": "A42",  # Мебель
            "savings": "A63",  # Средние сбережения
            "employment_duration": "A73",  # 1-4 года
            "personal_status": "A92",  # Женщина, разведена
            "debtors": "A102",  # Со-заемщик
            "property": "A122",  # Сберегательный договор
            "other_installment_plans": "A142",  # Магазин
            "housing": "A152",  # Собственное жилье
            "job": "A172",  # Неквалифицированный резидент
            "telephone": "A192",  # Есть телефон
            "foreign_worker": "A201"  # Гражданин
        })
        
        return edge_cases

class SpikeTestShape(LoadTestShape):
    """Профиль нагрузочного тестирования с пиковыми нагрузками"""
    
    stages = [
        # Длительность, пользователи, темп роста
        {"duration": 60, "users": 10, "spawn_rate": 1},    # Медленный рост
        {"duration": 120, "users": 50, "spawn_rate": 5},   # Средняя нагрузка
        {"duration": 60, "users": 200, "spawn_rate": 20},  # Пиковая нагрузка
        {"duration": 120, "users": 50, "spawn_rate": 5},   # Возврат к средней
        {"duration": 60, "users": 10, "spawn_rate": 1},    # Завершение
    ]
    
    def tick(self):
        """Определение текущей стадии тестирования"""
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data
        
        # Тест завершен
        return None

class SoakTestShape(LoadTestShape):
    """Профиль для длительного тестирования на стабильной нагрузке"""
    
    def tick(self):
        run_time = self.get_run_time()
        
        # 4-часовой soak test
        if run_time < 4 * 3600:  # 4 часа
            # Стабильная нагрузка: 100 пользователей
            return (100, 10)
        
        return None

class StressTestShape(LoadTestShape):
    """Профиль для стресс-тестирования с постепенным увеличением нагрузки"""
    
    def tick(self):
        run_time = self.get_run_time()
        
        # Постепенное увеличение нагрузки до предела
        if run_time < 300:  # 0-5 минут
            users = int(run_time / 300 * 50)  # До 50 пользователей
            return (max(1, users), 5)
        elif run_time < 600:  # 5-10 минут
            users = 50 + int((run_time - 300) / 300 * 100)  # До 150
            return (users, 10)
        elif run_time < 900:  # 10-15 минут
            users = 150 + int((run_time - 600) / 300 * 150)  # До 300
            return (users, 20)
        else:
            # Удержание максимальной нагрузки
            return (300, 20)

class EndpointMonitor:
    """Мониторинг endpoints во время нагрузочного тестирования"""
    
    def __init__(self):
        self.endpoint_stats = {}
        self.alert_thresholds = {
            'response_time': 1000,  # 1 секунда
            'error_rate': 5,        # 5%
            'availability': 99.9    # 99.9%
        }
    
    def record_request(self, endpoint, response_time, success):
        """Запись метрики запроса"""
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                'count': 0,
                'success_count': 0,
                'total_response_time': 0,
                'errors': 0
            }
        
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['total_response_time'] += response_time
        
        if success:
            stats['success_count'] += 1
        else:
            stats['errors'] += 1
        
        # Проверка алертов
        self._check_alerts(endpoint, stats)
    
    def _check_alerts(self, endpoint, stats):
        """Проверка условий для алертов"""
        if stats['count'] < 10:
            return
        
        avg_response_time = stats['total_response_time'] / stats['count']
        error_rate = (stats['errors'] / stats['count']) * 100
        availability = (stats['success_count'] / stats['count']) * 100
        
        alerts = []
        
        if avg_response_time > self.alert_thresholds['response_time']:
            alerts.append(
                f"High response time for {endpoint}: {avg_response_time:.0f}ms"
            )
        
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append(
                f"High error rate for {endpoint}: {error_rate:.1f}%"
            )
        
        if availability < self.alert_thresholds['availability']:
            alerts.append(
                f"Low availability for {endpoint}: {availability:.1f}%"
            )
        
        if alerts:
            logging.warning(f"Alerts for {endpoint}: {', '.join(alerts)}")
    
    def get_summary(self):
        """Получение сводной статистики"""
        summary = {
            'total_requests': 0,
            'total_errors': 0,
            'total_response_time': 0,
            'endpoints': {}
        }
        
        for endpoint, stats in self.endpoint_stats.items():
            summary['total_requests'] += stats['count']
            summary['total_errors'] += stats['errors']
            summary['total_response_time'] += stats['total_response_time']
            
            if stats['count'] > 0:
                avg_response_time = stats['total_response_time'] / stats['count']
                error_rate = (stats['errors'] / stats['count']) * 100
                availability = (stats['success_count'] / stats['count']) * 100
                
                summary['endpoints'][endpoint] = {
                    'request_count': stats['count'],
                    'average_response_time_ms': round(avg_response_time, 2),
                    'error_rate_percent': round(error_rate, 2),
                    'availability_percent': round(availability, 2)
                }
        
        if summary['total_requests'] > 0:
            summary['overall_availability'] = (
                (summary['total_requests'] - summary['total_errors']) / 
                summary['total_requests'] * 100
            )
            summary['overall_avg_response_time'] = (
                summary['total_response_time'] / summary['total_requests']
            )
        
        return summary

# Глобальный монитор
monitor = EndpointMonitor()

class MonitoredHttpUser(HttpUser):
    """Пользователь с мониторингом endpoints"""
    
    wait_time = between(1, 3)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "load-test-key"
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_generator = DataGenerator()
    
    @task(5)
    def monitored_single_prediction(self):
        data = self.data_generator.generate_single_application()
        
        start_time = datetime.now()
        
        with self.client.post(
            "/api/v1/predict",
            json=data,
            headers=self.headers,
            name="Monitored Single Prediction",
            catch_response=True
        ) as response:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            success = response.status_code == 200
            
            monitor.record_request(
                "single_prediction",
                response_time,
                success
            )
            
            if not success:
                response.failure(f"Status {response.status_code}")
            else:
                response.success()

# Команды для запуска тестов:
# 1. Обычный тест: locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 5m
# 2. Спайк тест: locust -f tests/load/locustfile.py --headless --host=http://localhost:8000
# 3. С параметрами формы: locust -f tests/load/locustfile.py --headless -u 50 -r 5 -t 10m --csv=results
# 4. Web интерфейс: locust -f tests/load/locustfile.py --web-host=0.0.0.0 --web-port=8089

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Запуск теста напрямую (для отладки)
        from locust import runners
        
        print("Starting load test...")
        
        # Можно добавить дополнительную логику здесь
        pass