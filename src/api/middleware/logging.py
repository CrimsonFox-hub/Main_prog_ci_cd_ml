"""
Middleware для логирования запросов
"""
import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import api_logger, RequestContext, get_request_context
import uuid

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware для логирования HTTP запросов"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Генерация уникального ID запроса
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Получение контекста запроса
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Начало отсчета времени
        start_time = time.time()
        
        # Установка контекста для логирования
        with RequestContext(
            request_id=request_id,
            correlation_id=request.headers.get("X-Correlation-ID", request_id),
            user_id=request.headers.get("X-User-ID", "")
        ):
            try:
                # Логирование входящего запроса
                api_logger.info(
                    f"Incoming request: {request.method} {request.url.path}",
                    extra={
                        "client_ip": client_ip,
                        "user_agent": user_agent,
                        "method": request.method,
                        "path": request.url.path,
                        "query_params": dict(request.query_params),
                        "headers": dict(request.headers)
                    }
                )
                
                # Обработка запроса
                response = await call_next(request)
                
                # Расчет времени выполнения
                duration = (time.time() - start_time) * 1000  # в миллисекундах
                
                # Логирование ответа
                api_logger.info(
                    f"Request completed: {request.method} {request.url.path}",
                    extra={
                        "status_code": response.status_code,
                        "duration_ms": duration,
                        "response_size": len(response.body) if hasattr(response, 'body') else 0,
                        "client_ip": client_ip,
                        "method": request.method,
                        "path": request.url.path
                    }
                )
                
                # Добавление заголовков для трассировки
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Duration-MS"] = str(duration)
                
                return response
                
            except Exception as e:
                # Логирование ошибок
                duration = (time.time() - start_time) * 1000
                
                api_logger.error(
                    f"Request failed: {request.method} {request.url.path}",
                    extra={
                        "error": str(e),
                        "duration_ms": duration,
                        "client_ip": client_ip,
                        "method": request.method,
                        "path": request.url.path,
                        "traceback": str(e.__traceback__)
                    },
                    exc_info=True
                )
                
                raise

class RequestBodyLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware для логирования тела запроса (с ограничениями)"""
    
    def __init__(self, app, max_body_size: int = 1024):  # 1KB максимум
        super().__init__(app)
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Логирование тела запроса для отладки
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            try:
                # Чтение тела запроса
                body = await request.body()
                
                # Ограничение размера для логирования
                if len(body) <= self.max_body_size:
                    try:
                        body_json = json.loads(body)
                        api_logger.debug(
                            f"Request body: {request.method} {request.url.path}",
                            extra={
                                "request_body": body_json,
                                "content_type": content_type,
                                "body_size": len(body)
                            }
                        )
                    except json.JSONDecodeError:
                        api_logger.debug(
                            f"Request body (not JSON): {request.method} {request.url.path}",
                            extra={
                                "body_preview": body[:100],
                                "content_type": content_type,
                                "body_size": len(body)
                            }
                        )
                else:
                    api_logger.debug(
                        f"Request body too large: {len(body)} bytes"
                    )
                
                # Восстановление тела запроса
                request._body = body
                
            except Exception as e:
                api_logger.warning(f"Failed to log request body: {e}")
        
        return await call_next(request)

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware для мониторинга производительности"""
    
    def __init__(self, app, slow_request_threshold: float = 1000):  # 1 секунда
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = (time.time() - start_time) * 1000
        
        # Логирование медленных запросов
        if duration > self.slow_request_threshold:
            api_logger.warning(
                f"Slow request detected: {request.method} {request.url.path}",
                extra={
                    "duration_ms": duration,
                    "threshold_ms": self.slow_request_threshold,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code
                }
            )
        
        # Добавление метрики в заголовок
        response.headers["X-Response-Time"] = f"{duration:.2f}ms"
        
        return response