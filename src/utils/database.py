"""
Утилиты для работы с базами данных
"""
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Iterator, Tuple
import logging
from datetime import datetime
import json
from decimal import Decimal
import time
from functools import wraps
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Типы баз данных"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"
    MONGODB = "mongodb"

@dataclass
class QueryStats:
    """Статистика выполнения запросов"""
    query: str
    execution_time: float
    rows_returned: int
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class ConnectionPool:
    """Пул соединений с базой данных"""
    
    def __init__(self, min_connections: int = 2, max_connections: int = 20,
                 host: str = "localhost", port: int = 5432,
                 database: str = "postgres", user: str = "postgres",
                 password: str = "", **kwargs):
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.kwargs = kwargs
        
        # Инициализация пула
        self.pool: Optional[pool.ThreadedConnectionPool] = None
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._lock = threading.Lock()
        self._initialize_pool()
        
        # Статистика
        self.stats: List[QueryStats] = []
        self.active_connections = 0
        self.total_queries = 0
        
    def _initialize_pool(self):
        """Инициализация пула соединений"""
        try:
            self.pool = pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                **self.kwargs
            )
            logger.info(f"Database connection pool initialized for {self.database}")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> Iterator[psycopg2.extensions.connection]:
        """Получение соединения из пула"""
        conn = None
        try:
            with self._lock:
                conn = self.pool.getconn()
                self.active_connections += 1
            
            yield conn
            conn.commit()
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
            
        finally:
            if conn:
                with self._lock:
                    self.pool.putconn(conn)
                    self.active_connections -= 1
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = True) -> Iterator[psycopg2.extensions.cursor]:
        """Получение курсора из соединения"""
        with self.get_connection() as conn:
            cursor_class = RealDictCursor if dict_cursor else psycopg2.extensions.cursor
            cursor = conn.cursor(cursor_factory=cursor_class)
            
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: Tuple = None, 
                     fetch: bool = True, dict_cursor: bool = True) -> List[Dict[str, Any]]:
        """Выполнение SQL запроса"""
        start_time = time.time()
        stats = QueryStats(
            query=query,
            execution_time=0,
            rows_returned=0,
            timestamp=datetime.now(),
            success=False
        )
        
        try:
            with self.get_cursor(dict_cursor) as cursor:
                cursor.execute(query, params)
                
                if fetch:
                    if query.strip().upper().startswith('SELECT'):
                        result = cursor.fetchall()
                        stats.rows_returned = len(result)
                    else:
                        result = cursor.rowcount
                else:
                    result = cursor.rowcount
                
                stats.execution_time = time.time() - start_time
                stats.success = True
                
                self.total_queries += 1
                self.stats.append(stats)
                
                logger.debug(f"Query executed in {stats.execution_time:.3f}s: {query[:100]}...")
                
                return result
                
        except Exception as e:
            stats.execution_time = time.time() - start_time
            stats.error_message = str(e)
            self.stats.append(stats)
            
            logger.error(f"Query failed: {e}\nQuery: {query}")
            raise
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Выполнение множества запросов"""
        start_time = time.time()
        total_rows = 0
        
        try:
            with self.get_cursor(dict_cursor=False) as cursor:
                for params in params_list:
                    cursor.execute(query, params)
                    total_rows += cursor.rowcount
                
                execution_time = time.time() - start_time
                logger.info(f"Executed {len(params_list)} queries in {execution_time:.3f}s")
                
                return total_rows
                
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise
    
    def execute_values(self, query: str, values: List[Tuple], 
                      template: str = None, page_size: int = 100) -> int:
        """Эффективная вставка множества значений"""
        start_time = time.time()
        
        try:
            with self.get_cursor(dict_cursor=False) as cursor:
                execute_values(cursor, query, values, template=template, page_size=page_size)
                affected_rows = cursor.rowcount
                
                execution_time = time.time() - start_time
                logger.info(f"Inserted {affected_rows} rows in {execution_time:.3f}s")
                
                return affected_rows
                
        except Exception as e:
            logger.error(f"execute_values failed: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Получение информации о таблице"""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        
        columns = self.execute_query(query, (table_name,))
        
        # Получение индексов
        index_query = """
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE tablename = %s
        """
        
        indexes = self.execute_query(index_query, (table_name,))
        
        return {
            'table_name': table_name,
            'columns': columns,
            'indexes': indexes
        }
    
    def backup_table(self, table_name: str, backup_suffix: str = None) -> str:
        """Создание backup таблицы"""
        if not backup_suffix:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_table = f"{table_name}_backup_{backup_suffix}"
        
        # Создание backup таблицы
        create_query = f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}"
        self.execute_query(create_query, fetch=False)
        
        # Создание индексов
        index_query = f"""
        SELECT indexdef 
        FROM pg_indexes 
        WHERE tablename = '{table_name}'
        """
        
        indexes = self.execute_query(index_query)
        for index in indexes:
            index_def = index['indexdef']
            # Замена имени таблицы в определении индекса
            new_index_def = index_def.replace(table_name, backup_table)
            self.execute_query(new_index_def, fetch=False)
        
        logger.info(f"Backup created: {backup_table}")
        return backup_table
    
    def restore_table(self, table_name: str, backup_table: str) -> None:
        """Восстановление таблицы из backup"""
        # Удаление текущей таблицы
        self.execute_query(f"DROP TABLE IF EXISTS {table_name} CASCADE", fetch=False)
        
        # Создание таблицы из backup
        self.execute_query(f"CREATE TABLE {table_name} AS SELECT * FROM {backup_table}", fetch=False)
        
        # Восстановление индексов
        index_query = f"""
        SELECT indexdef 
        FROM pg_indexes 
        WHERE tablename = '{backup_table}'
        """
        
        indexes = self.execute_query(index_query)
        for index in indexes:
            index_def = index['indexdef']
            # Замена имени таблицы в определении индекса
            new_index_def = index_def.replace(backup_table, table_name)
            self.execute_query(new_index_def, fetch=False)
        
        logger.info(f"Table restored from backup: {backup_table}")
    
    def get_performance_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение статистики производительности запросов"""
        # Агрегация статистики
        stats_summary = {}
        
        for stat in self.stats[-limit:]:
            query_key = stat.query[:100]  # Берем первые 100 символов как ключ
            
            if query_key not in stats_summary:
                stats_summary[query_key] = {
                    'query_pattern': query_key,
                    'count': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'errors': 0,
                    'last_executed': stat.timestamp
                }
            
            summary = stats_summary[query_key]
            summary['count'] += 1
            summary['total_time'] += stat.execution_time
            
            if not stat.success:
                summary['errors'] += 1
            
            if stat.timestamp > summary['last_executed']:
                summary['last_executed'] = stat.timestamp
        
        # Расчет среднего времени
        for summary in stats_summary.values():
            if summary['count'] > 0:
                summary['avg_time'] = summary['total_time'] / summary['count']
        
        # Сортировка по общему времени выполнения
        sorted_stats = sorted(
            stats_summary.values(),
            key=lambda x: x['total_time'],
            reverse=True
        )
        
        return sorted_stats
    
    def close(self):
        """Закрытие пула соединений"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Декоратор для повторных попыток при ошибках БД"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    last_exception = e
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Экспоненциальная задержка
                    else:
                        logger.error(f"Max retries reached for {func.__name__}")
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator

class DatabaseManager:
    """Менеджер для работы с несколькими базами данных"""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.default_pool_name = "default"
    
    def add_pool(self, name: str, **kwargs) -> ConnectionPool:
        """Добавление пула соединений"""
        pool = ConnectionPool(**kwargs)
        self.pools[name] = pool
        return pool
    
    def get_pool(self, name: str = None) -> ConnectionPool:
        """Получение пула соединений"""
        pool_name = name or self.default_pool_name
        
        if pool_name not in self.pools:
            raise ValueError(f"Pool '{pool_name}' not found")
        
        return self.pools[pool_name]
    
    def execute_in_transaction(self, pool_name: str, queries: List[Tuple[str, Tuple]]) -> List[Any]:
        """Выполнение нескольких запросов в транзакции"""
        pool = self.get_pool(pool_name)
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            results = []
            
            try:
                for query, params in queries:
                    cursor.execute(query, params)
                    
                    if query.strip().upper().startswith('SELECT'):
                        results.append(cursor.fetchall())
                    else:
                        results.append(cursor.rowcount)
                
                conn.commit()
                return results
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction failed: {e}")
                raise
    
    def migrate_database(self, migration_scripts: List[str], pool_name: str = None) -> None:
        """Применение миграций к базе данных"""
        pool = self.get_pool(pool_name)
        
        # Создание таблицы для отслеживания миграций
        create_migrations_table = """
        CREATE TABLE IF NOT EXISTS migrations (
            id SERIAL PRIMARY KEY,
            script_name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64),
            success BOOLEAN DEFAULT TRUE
        )
        """
        
        pool.execute_query(create_migrations_table, fetch=False)
        
        # Получение уже примененных миграций
        applied_migrations = pool.execute_query(
            "SELECT script_name FROM migrations WHERE success = TRUE"
        )
        applied_names = {m['script_name'] for m in applied_migrations}
        
        # Применение новых миграций
        for script_path in migration_scripts:
            script_name = script_path.split('/')[-1]
            
            if script_name in applied_names:
                logger.info(f"Migration already applied: {script_name}")
                continue
            
            try:
                with open(script_path, 'r') as f:
                    sql_script = f.read()
                
                # Выполнение скрипта
                pool.execute_query(sql_script, fetch=False)
                
                # Запись о примененной миграции
                pool.execute_query(
                    "INSERT INTO migrations (script_name) VALUES (%s)",
                    (script_name,),
                    fetch=False
                )
                
                logger.info(f"Migration applied successfully: {script_name}")
                
            except Exception as e:
                logger.error(f"Migration failed: {script_name} - {e}")
                
                # Запись о неудачной миграции
                pool.execute_query(
                    "INSERT INTO migrations (script_name, success) VALUES (%s, %s)",
                    (script_name, False),
                    fetch=False
                )
                raise
    
    def close_all(self):
        """Закрытие всех пулов соединений"""
        for name, pool in self.pools.items():
            try:
                pool.close()
                logger.info(f"Closed pool: {name}")
            except Exception as e:
                logger.error(f"Error closing pool {name}: {e}")

# Глобальный инстанс менеджера БД
_db_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Получение глобального инстанса менеджера БД"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def init_database_from_config(config: Dict[str, Any]) -> ConnectionPool:
    """Инициализация БД из конфигурации"""
    manager = get_database_manager()
    
    pool = manager.add_pool(
        "default",
        host=config['host'],
        port=config['port'],
        database=config['database'],
        user=config['username'],
        password=config['password'],
        min_connections=config.get('pool_size', 2),
        max_connections=config.get('max_overflow', 20)
    )
    
    return pool

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Пример конфигурации
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'credit_scoring',
        'username': 'postgres',
        'password': 'password',
        'pool_size': 5,
        'max_overflow': 10
    }
    
    try:
        # Инициализация БД
        pool = init_database_from_config(db_config)
        
        # Пример запроса
        @retry_on_failure(max_retries=3)
        def get_prediction_count():
            result = pool.execute_query("SELECT COUNT(*) as count FROM predictions")
            return result[0]['count']
        
        count = get_prediction_count()
        print(f"Total predictions: {count}")
        
        # Получение статистики производительности
        stats = pool.get_performance_stats(limit=10)
        print("\nTop 10 queries by execution time:")
        for stat in stats:
            print(f"  {stat['query_pattern']}: {stat['avg_time']:.3f}s avg")
        
        # Закрытие соединений
        get_database_manager().close_all()
        
    except Exception as e:
        print(f"Database error: {e}")