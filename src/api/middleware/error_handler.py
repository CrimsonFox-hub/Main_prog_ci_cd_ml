"""
Middleware для обработки ошибок
"""
import traceback
from typing import Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import api_logger, get_request_context

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware для обработки исключений"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
            
        except HTTPException as http_exc:
            # Обработка HTTP исключений
            return await self.handle_http_exception(request, http_exc)
            
        except RequestValidationError as validation_exc:
            # Обработка ошибок валидации
            return await self.handle_validation_error(request, validation_exc)
            
        except Exception as exc:
            # Обработка неожиданных исключений
            return await self.handle_unexpected_error(request, exc)
    
    async def handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Обработка HTTP исключений"""
        context = get_request_context()
        
        api_logger.warning(
            f"HTTP exception: {exc.status_code}",
            extra={
                "status_code": exc.status_code,
                "detail": exc.detail,
                "method": request.method,
                "path": request.url.path,
                "request_id": context.get('request_id', ''),
                "headers": dict(request.headers)
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "request_id": context.get('request_id', ''),
                    "type": "http_error"
                }
            },
            headers={
                "X-Request-ID": context.get('request_id', ''),
                "X-Error-Type": "http_error"
            }
        )
    
    async def handle_validation_error(self, request: Request, exc: RequestValidationError) -> JSONResponse:
        """Обработка ошибок валидации"""
        context = get_request_context()
        
        # Форматирование ошибок валидации
        errors = []
        for error in exc.errors():
            errors.append({
                "field": " -> ".join([str(loc) for loc in error.get("loc", [])]),
                "message": error.get("msg"),
                "type": error.get("type")
            })
        
        api_logger.warning(
            f"Validation error: {len(errors)} errors",
            extra={
                "errors": errors,
                "method": request.method,
                "path": request.url.path,
                "request_id": context.get('request_id', ''),
                "body": exc.body if hasattr(exc, 'body') else None
            }
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": 422,
                    "message": "Validation error",
                    "errors": errors,
                    "request_id": context.get('request_id', ''),
                    "type": "validation_error"
                }
            },
            headers={
                "X-Request-ID": context.get('request_id', ''),
                "X-Error-Type": "validation_error"
            }
        )
    
    async def handle_unexpected_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Обработка неожиданных исключений"""
        context = get_request_context()
        
        # Получение трейсбэка
        exc_traceback = traceback.format_exc()
        
        api_logger.error(
            f"Unexpected error: {type(exc).__name__}",
            extra={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": exc_traceback,
                "method": request.method,
                "path": request.url.path,
                "request_id": context.get('request_id', ''),
                "client_ip": request.client.host if request.client else "unknown"
            },
            exc_info=True
        )
        
        # В production не показываем детали ошибки
        is_production = get_config('api', 'environment') == 'production'
        
        error_detail = "Internal server error" if is_production else str(exc)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": error_detail,
                    "request_id": context.get('request_id', ''),
                    "type": "server_error"
                }
            },
            headers={
                "X-Request-ID": context.get('request_id', ''),
                "X-Error-Type": "server_error"
            }
        )

class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Middleware для реализации Circuit Breaker паттерна"""
    
    def __init__(self, app):
        super().__init__(app)
        self.failure_count = 0
        self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
        self.reset_timeout = 60  # секунды
        
    async def dispatch(self, request: Request, call_next):
        # Проверка состояния circuit breaker
        if self.circuit_state == "OPEN":
            time_since_failure = time.time() - self.last_failure_time
            
            if time_since_failure > self.reset_timeout:
                self.circuit_state = "HALF_OPEN"
                api_logger.info("Circuit breaker state changed to HALF_OPEN")
            else:
                api_logger.warning("Circuit breaker is OPEN, rejecting request")
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "code": 503,
                            "message": "Service temporarily unavailable",
                            "type": "circuit_breaker_open"
                        }
                    }
                )
        
        try:
            response = await call_next(request)
            
            # Сброс счетчика при успешном запросе
            if self.circuit_state == "HALF_OPEN":
                self.circuit_state = "CLOSED"
                self.failure_count = 0
                api_logger.info("Circuit breaker state changed to CLOSED")
            
            return response
            
        except Exception as exc:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Проверка условий для открытия circuit breaker
            if self.failure_count >= 5:  # порог сбоев
                self.circuit_state = "OPEN"
                api_logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise exc