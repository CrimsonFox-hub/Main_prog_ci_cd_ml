"""
Middleware для ограничения частоты запросов
"""
import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import api_logger
from src.utils.config_loader import get_config

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Middleware для ограничения частоты запросов"""
    
    def __init__(self, app):
        super().__init__(app)
        self.config = get_config('api')
        self.rate_limit_config = self.config.get('rate_limiting', {})
        
        # Хранилище для отслеживания запросов
        self.request_logs: Dict[str, list] = {}
        
        # Настройки по умолчанию
        self.default_limit = self.rate_limit_config.get('default_limit', 60)  # запросов в минуту
        self.burst_limit = self.rate_limit_config.get('burst_limit', 10)  # burst requests
        self.window_size = 60  # секунды
    
    def get_client_identifier(self, request: Request) -> str:
        """Получение идентификатора клиента"""
        # Используем IP адрес или API ключ для идентификации
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def is_rate_limited(self, client_id: str, endpoint: str) -> Tuple[bool, Dict[str, int]]:
        """Проверка, превышен ли лимит запросов"""
        current_time = time.time()
        key = f"{client_id}:{endpoint}"
        
        # Инициализация логов для клиента
        if key not in self.request_logs:
            self.request_logs[key] = []
        
        # Очистка старых записей
        self.request_logs[key] = [
            req_time for req_time in self.request_logs[key]
            if current_time - req_time < self.window_size
        ]
        
        # Проверка лимитов
        request_count = len(self.request_logs[key])
        
        if request_count >= self.default_limit:
            # Превышен основной лимит
            return True, {
                "limit": self.default_limit,
                "remaining": 0,
                "reset_in": int(self.window_size - (current_time - self.request_logs[key][0]))
            }
        
        # Проверка burst лимита (за последние 10 секунд)
        recent_requests = [
            req_time for req_time in self.request_logs[key]
            if current_time - req_time < 10
        ]
        
        if len(recent_requests) >= self.burst_limit:
            # Превышен burst лимит
            return True, {
                "limit": self.burst_limit,
                "remaining": 0,
                "reset_in": int(10 - (current_time - recent_requests[0]))
            }
        
        # Добавление текущего запроса
        self.request_logs[key].append(current_time)
        
        # Ограничение размера логов
        if len(self.request_logs[key]) > 1000:
            self.request_logs[key] = self.request_logs[key][-100:]
        
        return False, {
            "limit": self.default_limit,
            "remaining": self.default_limit - request_count - 1,
            "reset_in": int(self.window_size - (current_time - self.request_logs[key][0]) if self.request_logs[key] else self.window_size)
        }
    
    async def dispatch(self, request: Request, call_next):
        # Получение идентификатора клиента
        client_id = self.get_client_identifier(request)
        endpoint = request.url.path
        
        # Проверка ограничений
        is_limited, rate_info = self.is_rate_limited(client_id, endpoint)
        
        if is_limited:
            api_logger.warning(
                f"Rate limit exceeded for {client_id} on {endpoint}",
                extra={
                    "client_id": client_id,
                    "endpoint": endpoint,
                    "rate_info": rate_info
                }
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": rate_info["limit"],
                    "reset_in": rate_info["reset_in"]
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset_in"])
                }
            )
        
        # Добавление заголовков с информацией о лимитах
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset_in"])
        
        return response

class AdaptiveRateLimiter(RateLimiterMiddleware):
    """Адаптивный rate limiter с динамическими лимитами"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Динамические лимиты в зависимости от нагрузки
        self.load_factors = {
            'low': 1.5,    # Увеличиваем лимит на 50%
            'normal': 1.0, # Стандартный лимит
            'high': 0.5,   # Уменьшаем лимит на 50%
            'critical': 0.2 # Уменьшаем лимит на 80%
        }
        
        # Мониторинг нагрузки
        self.request_times = []
        self.window_size = 30  # секунды
    
    def get_current_load_factor(self) -> float:
        """Расчет текущего коэффициента нагрузки"""
        current_time = time.time()
        
        # Очистка старых записей
        self.request_times = [
            req_time for req_time in self.request_times
            if current_time - req_time < self.window_size
        ]
        
        # Расчет нагрузки (запросов в секунду)
        requests_per_second = len(self.request_times) / self.window_size
        
        # Определение уровня нагрузки
        if requests_per_second < 10:
            return self.load_factors['low']
        elif requests_per_second < 50:
            return self.load_factors['normal']
        elif requests_per_second < 100:
            return self.load_factors['high']
        else:
            return self.load_factors['critical']
    
    def get_adjusted_limit(self, base_limit: int) -> int:
        """Получение скорректированного лимита на основе нагрузки"""
        load_factor = self.get_current_load_factor()
        return int(base_limit * load_factor)
    
    async def dispatch(self, request: Request, call_next):
        # Обновление мониторинга нагрузки
        self.request_times.append(time.time())
        
        # Ограничение размера
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-100:]
        
        # Динамическая корректировка лимитов
        adjusted_limit = self.get_adjusted_limit(self.default_limit)
        adjusted_burst = self.get_adjusted_limit(self.burst_limit)
        
        # Временная замена лимитов
        original_limit = self.default_limit
        original_burst = self.burst_limit
        
        self.default_limit = adjusted_limit
        self.burst_limit = adjusted_burst
        
        try:
            response = await super().dispatch(request, call_next)
            
            # Добавление информации о нагрузке
            response.headers["X-Load-Factor"] = str(self.get_current_load_factor())
            
            return response
            
        finally:
            # Восстановление оригинальных лимитов
            self.default_limit = original_limit
            self.burst_limit = original_burst