"""
Утилиты для логирования с поддержкой структурированного логирования
"""
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import socket
import threading
from contextvars import ContextVar
import yaml

# Контекстные переменные для трассировки запросов
request_id: ContextVar[str] = ContextVar('request_id', default='')
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')
user_id: ContextVar[str] = ContextVar('user_id', default='')

class StructuredFormatter(logging.Formatter):
    """Форматтер для структурированного логирования в JSON"""
    
    def __init__(self, include_hostname: bool = True):
        super().__init__()
        self.hostname = socket.gethostname() if include_hostname else None
        self.pid = None
        
    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи лога в JSON"""
        # Базовые поля
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.threadName,
            'process_id': record.process,
            'process_name': record.processName
        }
        
        # Добавление хоста
        if self.hostname:
            log_entry['hostname'] = self.hostname
        
        # Добавление контекстных переменных
        try:
            req_id = request_id.get()
            if req_id:
                log_entry['request_id'] = req_id
        except LookupError:
            pass
        
        try:
            corr_id = correlation_id.get()
            if corr_id:
                log_entry['correlation_id'] = corr_id
        except LookupError:
            pass
        
        try:
            uid = user_id.get()
            if uid:
                log_entry['user_id'] = uid
        except LookupError:
            pass
        
        # Добавление эксепшнов
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Добавление дополнительных атрибутов
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        # Добавление времени выполнения для performance логирования
        if hasattr(record, 'duration'):
            log_entry['duration_ms'] = record.duration
        
        return json.dumps(log_entry, ensure_ascii=False)

class PerformanceFilter(logging.Filter):
    """Фильтр для performance логирования"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, 'duration'):
            return True
        
        # Фильтрация быстрых операций
        return record.duration > 100  # Логируем только операции > 100ms

class RequestContextFilter(logging.Filter):
    """Фильтр для добавления контекста запроса"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Добавление контекстных переменных
        try:
            record.request_id = request_id.get()
        except LookupError:
            record.request_id = ''
        
        try:
            record.correlation_id = correlation_id.get()
        except LookupError:
            record.correlation_id = ''
        
        try:
            record.user_id = user_id.get()
        except LookupError:
            record.user_id = ''
        
        return True

class TimedContextManager:
    """Контекстный менеджер для измерения времени выполнения"""
    
    def __init__(self, name: str, logger: logging.Logger, level: int = logging.INFO):
        self.name = name
        self.logger = logger
        self.level = level
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000  # в миллисекундах
        
        # Создание записи лога с duration
        if self.logger.isEnabledFor(self.level):
            record = self.logger.makeRecord(
                self.logger.name,
                self.level,
                self.name,
                0,
                f"Operation '{self.name}' completed",
                (),
                None,
                None,
                extra={'duration': duration}
            )
            self.logger.handle(record)

class LoggerFactory:
    """Фабрика для создания логгеров с конфигурацией"""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, name: str, config_path: Optional[str] = None) -> logging.Logger:
        """Получение или создание логгера"""
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Загрузка конфигурации
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Конфигурация по умолчанию
            config = {
                'level': 'INFO',
                'format': 'json',
                'handlers': ['console', 'file'],
                'file_path': 'logs/app.log',
                'max_size_mb': 100,
                'backup_count': 10
            }
        
        # Создание логгера
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.get('level', 'INFO')))
        logger.propagate = False
        
        # Удаление существующих обработчиков
        logger.handlers.clear()
        
        # Добавление фильтра контекста
        context_filter = RequestContextFilter()
        logger.addFilter(context_filter)
        
        # Настройка обработчиков
        handlers = config.get('handlers', ['console'])
        
        if 'console' in handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if config.get('format') == 'json':
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if 'file' in handlers:
            file_path = Path(config.get('file_path', 'logs/app.log'))
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=config.get('max_size_mb', 100) * 1024 * 1024,
                backupCount=config.get('backup_count', 10)
            )
            
            if config.get('format') == 'json':
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Добавление performance фильтра для файлового логгера
        if 'performance' in handlers:
            perf_filter = PerformanceFilter()
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.addFilter(perf_filter)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def configure_from_dict(cls, config: Dict[str, Any]):
        """Конфигурация всех логгеров из словаря"""
        for name, logger_config in config.get('loggers', {}).items():
            logger = cls.get_logger(name)
            
            if 'level' in logger_config:
                logger.setLevel(getattr(logging, logger_config['level']))
    
    @classmethod
    def shutdown(cls):
        """Завершение работы всех логгеров"""
        for logger in cls._loggers.values():
            for handler in logger.handlers:
                handler.close()
        cls._loggers.clear()

def setup_logging(config_path: Optional[str] = None) -> None:
    """Настройка системы логирования"""
    # Загрузка конфигурации
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get('root_level', 'WARNING')))
    
    # Удаление стандартных обработчиков
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Создание обработчика для корневого логгера
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.get('console_level', 'INFO')))
    
    if config.get('format') == 'json':
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Настройка логгеров из конфигурации
    LoggerFactory.configure_from_dict(config)

def log_execution_time(logger: logging.Logger, level: int = logging.INFO):
    """Декоратор для логирования времени выполнения функции"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = (time.time() - start_time) * 1000
                
                if logger.isEnabledFor(level):
                    record = logger.makeRecord(
                        logger.name,
                        level,
                        func.__name__,
                        0,
                        f"Function '{func.__name__}' executed",
                        (),
                        None,
                        None,
                        extra={'duration': duration}
                    )
                    logger.handle(record)
        
        return wrapper
    return decorator

class RequestContext:
    """Контекстный менеджер для установки контекста запроса"""
    
    def __init__(self, request_id: str, correlation_id: Optional[str] = None, 
                 user_id: Optional[str] = None):
        self.request_id = request_id
        self.correlation_id = correlation_id or request_id
        self.user_id = user_id
        self.token1 = None
        self.token2 = None
        self.token3 = None
        
    def __enter__(self):
        # Установка контекстных переменных
        self.token1 = request_id.set(self.request_id)
        self.token2 = correlation_id.set(self.correlation_id)
        if self.user_id:
            self.token3 = user_id.set(self.user_id)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Восстановление предыдущих значений
        if self.token1:
            request_id.reset(self.token1)
        if self.token2:
            correlation_id.reset(self.token2)
        if self.token3:
            user_id.reset(self.token3)

def get_request_context() -> Dict[str, str]:
    """Получение текущего контекста запроса"""
    context = {}
    
    try:
        context['request_id'] = request_id.get()
    except LookupError:
        context['request_id'] = ''
    
    try:
        context['correlation_id'] = correlation_id.get()
    except LookupError:
        context['correlation_id'] = ''
    
    try:
        context['user_id'] = user_id.get()
    except LookupError:
        context['user_id'] = ''
    
    return context

# Экспорт основных логгеров
api_logger = LoggerFactory.get_logger('api')
model_logger = LoggerFactory.get_logger('model')
monitoring_logger = LoggerFactory.get_logger('monitoring')
database_logger = LoggerFactory.get_logger('database')
cache_logger = LoggerFactory.get_logger('cache')

if __name__ == "__main__":
    # Пример использования
    setup_logging('configs/logging_config.yaml')
    
    logger = LoggerFactory.get_logger('test')
    
    # Логирование с контекстом
    with RequestContext('req-123', 'corr-456', 'user-789'):
        logger.info("Processing request")
        
        # Логирование времени выполнения
        with TimedContextManager('data_processing', logger):
            time.sleep(0.1)
        
        # Использование декоратора
        @log_execution_time(logger)
        def process_data(data):
            time.sleep(0.05)
            return len(data)
        
        process_data([1, 2, 3, 4, 5])
    
    # Завершение работы
    LoggerFactory.shutdown()