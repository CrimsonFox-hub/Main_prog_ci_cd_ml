"""
Утилита для загрузки и управления конфигурациями
"""
import yaml
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

class Environment(str, Enum):
    """Доступные окружения"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Конфигурация базы данных"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    @property
    def connection_string(self) -> str:
        """Получение строки подключения"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    name: str
    version: str
    path: str
    input_shape: tuple
    output_shape: tuple
    framework: str = "onnx"
    quantization: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Создание из словаря"""
        input_shape = tuple(data['input_shape']) if isinstance(data['input_shape'], list) else data['input_shape']
        output_shape = tuple(data['output_shape']) if isinstance(data['output_shape'], list) else data['output_shape']
        
        return cls(
            name=data['name'],
            version=data['version'],
            path=data['path'],
            input_shape=input_shape,
            output_shape=output_shape,
            framework=data.get('framework', 'onnx'),
            quantization=data.get('quantization', False)
        )

@dataclass
class APIConfig:
    """Конфигурация API"""
    host: str
    port: int
    workers: int
    debug: bool
    log_level: str
    cors_origins: list = field(default_factory=list)
    rate_limit: int = 60
    request_timeout: int = 30
    
    @property
    def server_url(self) -> str:
        """Получение URL сервера"""
        return f"http://{self.host}:{self.port}"

class ConfigLoader:
    """Загрузчик конфигураций с поддержкой окружений и секретов"""
    
    def __init__(self, config_dir: str = "configs", environment: Optional[str] = None):
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self._configs: Dict[str, Any] = {}
        self._secrets: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_name: str, use_environment: bool = True) -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            config_path = self.config_dir / f"{config_name}.yml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Загрузка базовой конфигурации
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Загрузка конфигурации окружения
        if use_environment:
            env_config_path = self.config_dir / f"{config_name}_{self.environment}.yaml"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f) or {}
                    config = self._merge_configs(config, env_config)
        
        # Замена переменных окружения
        config = self._replace_env_variables(config)
        
        # Загрузка секретов
        config = self._load_secrets(config)
        
        self._configs[config_name] = config
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Рекурсивное объединение конфигураций"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _replace_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Замена переменных окружения в конфигурации"""
        if isinstance(config, dict):
            return {k: self._replace_env_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_variables(v) for v in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            default_value = None
            
            # Проверка на наличие значения по умолчанию
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)
            
            value = os.getenv(env_var, default_value)
            
            if value is None:
                self.logger.warning(f"Environment variable {env_var} not found")
                return config
            
            # Преобразование типов
            if default_value is not None:
                if default_value.lower() in ('true', 'false'):
                    return value.lower() == 'true'
                elif default_value.isdigit():
                    return int(value)
                elif self._is_float(default_value):
                    return float(value)
            
            return value
        else:
            return config
    
    def _is_float(self, value: str) -> bool:
        """Проверка, является ли строка float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _load_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Загрузка секретов из файлов или vault"""
        secrets_path = Path('secrets')
        
        if not secrets_path.exists():
            return config
        
        # Загрузка секретов из файлов
        for secret_file in secrets_path.glob('*.secret'):
            secret_name = secret_file.stem
            with open(secret_file, 'r') as f:
                secret_value = f.read().strip()
                self._secrets[secret_name] = secret_value
        
        # Замена секретов в конфигурации
        return self._replace_secrets(config)
    
    def _replace_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Замена ссылок на секреты в конфигурации"""
        if isinstance(config, dict):
            return {k: self._replace_secrets(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_secrets(v) for v in config]
        elif isinstance(config, str) and config.startswith('secret://'):
            secret_name = config[9:]
            return self._secrets.get(secret_name, config)
        else:
            return config
    
    def get(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """Получение значения конфигурации"""
        if config_name not in self._configs:
            self.load_config(config_name)
        
        config = self._configs[config_name]
        
        if key is None:
            return config
        
        # Поиск по пути с точками
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_database_config(self) -> DatabaseConfig:
        """Получение конфигурации базы данных"""
        db_config = self.get('database', 'postgres')
        
        return DatabaseConfig(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            username=db_config['username'],
            password=db_config['password'],
            pool_size=db_config.get('pool_size', 10),
            max_overflow=db_config.get('max_overflow', 20),
            pool_timeout=db_config.get('pool_timeout', 30),
            pool_recycle=db_config.get('pool_recycle', 3600)
        )
    
    def get_model_config(self) -> ModelConfig:
        """Получение конфигурации модели"""
        model_config = self.get('model')
        
        return ModelConfig.from_dict(model_config)
    
    def get_api_config(self) -> APIConfig:
        """Получение конфигурации API"""
        api_config = self.get('api', 'server')
        
        return APIConfig(
            host=api_config['host'],
            port=api_config['port'],
            workers=api_config['workers'],
            debug=api_config.get('debug', False),
            log_level=api_config.get('log_level', 'INFO'),
            cors_origins=api_config.get('cors_origins', []),
            rate_limit=api_config.get('rate_limit', 60),
            request_timeout=api_config.get('request_timeout', 30)
        )
    
    def save_config(self, config_name: str, config: Dict[str, Any], 
                   environment: Optional[str] = None) -> None:
        """Сохранение конфигурации в файл"""
        if environment:
            config_path = self.config_dir / f"{config_name}_{environment}.yaml"
        else:
            config_path = self.config_dir / f"{config_name}.yaml"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Config saved to: {config_path}")
    
    def validate_config(self, config_name: str, schema: Dict[str, Any]) -> bool:
        """Валидация конфигурации по схеме"""
        config = self.get(config_name)
        
        def validate_section(section, schema_section, path=""):
            errors = []
            
            if isinstance(schema_section, dict):
                for key, value_schema in schema_section.items():
                    full_path = f"{path}.{key}" if path else key
                    
                    if key not in section:
                        if 'required' in value_schema and value_schema['required']:
                            errors.append(f"Missing required key: {full_path}")
                        continue
                    
                    if isinstance(value_schema, dict) and 'type' in value_schema:
                        expected_type = value_schema['type']
                        actual_value = section[key]
                        
                        if expected_type == 'string' and not isinstance(actual_value, str):
                            errors.append(f"{full_path}: expected string, got {type(actual_value)}")
                        elif expected_type == 'integer' and not isinstance(actual_value, int):
                            errors.append(f"{full_path}: expected integer, got {type(actual_value)}")
                        elif expected_type == 'number' and not isinstance(actual_value, (int, float)):
                            errors.append(f"{full_path}: expected number, got {type(actual_value)}")
                        elif expected_type == 'boolean' and not isinstance(actual_value, bool):
                            errors.append(f"{full_path}: expected boolean, got {type(actual_value)}")
                        elif expected_type == 'list' and not isinstance(actual_value, list):
                            errors.append(f"{full_path}: expected list, got {type(actual_value)}")
                        elif expected_type == 'dict' and not isinstance(actual_value, dict):
                            errors.append(f"{full_path}: expected dict, got {type(actual_value)}")
                    
                    # Рекурсивная валидация
                    if isinstance(value_schema, dict) and isinstance(section.get(key), dict):
                        sub_errors = validate_section(section[key], value_schema, full_path)
                        errors.extend(sub_errors)
            
            return errors
        
        errors = validate_section(config, schema)
        
        if errors:
            self.logger.error(f"Config validation failed for {config_name}:")
            for error in errors:
                self.logger.error(f"  - {error}")
            return False
        
        self.logger.info(f"Config validation passed for {config_name}")
        return True
    
    def watch_config(self, config_name: str, callback) -> None:
        """Наблюдение за изменениями конфигурации"""
        import time
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class ConfigChangeHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if config_name in event.src_path:
                    self.logger.info(f"Config file changed: {event.src_path}")
                    try:
                        # Перезагрузка конфигурации
                        self.load_config(config_name)
                        callback(self.get(config_name))
                    except Exception as e:
                        self.logger.error(f"Failed to reload config: {e}")
        
        handler = ConfigChangeHandler()
        handler.logger = self.logger
        
        observer = Observer()
        observer.schedule(handler, str(self.config_dir), recursive=False)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        
        observer.join()

# Схема для валидации конфигурации API
API_CONFIG_SCHEMA = {
    'server': {
        'host': {'type': 'string', 'required': True},
        'port': {'type': 'integer', 'required': True},
        'workers': {'type': 'integer', 'required': True},
        'debug': {'type': 'boolean', 'required': False},
        'log_level': {'type': 'string', 'required': False}
    }
}

# Глобальный инстанс загрузчика конфигураций
_config_loader: Optional[ConfigLoader] = None

def get_config_loader() -> ConfigLoader:
    """Получение глобального инстанса загрузчика конфигураций"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def get_config(config_name: str, key: str = None, default: Any = None) -> Any:
    """Получение конфигурации (глобальная функция)"""
    return get_config_loader().get(config_name, key, default)

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание загрузчика
    loader = ConfigLoader(environment="development")
    
    # Загрузка конфигурации
    try:
        api_config = loader.load_config('api')
        print("API Config:", json.dumps(api_config, indent=2))
        
        # Получение конкретных значений
        server_host = loader.get('api', 'server.host')
        print(f"Server host: {server_host}")
        
        # Получение структурированных конфигураций
        db_config = loader.get_database_config()
        print(f"Database connection string: {db_config.connection_string}")
        
        # Валидация конфигурации
        is_valid = loader.validate_config('api', API_CONFIG_SCHEMA)
        print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")
        
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)