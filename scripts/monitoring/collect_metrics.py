"""
Сбор метрик для мониторинга
"""
import time
import requests
import psutil
import json
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import Dict, List, Any
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.metrics = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        default_config = {
            'collection_interval': 30,
            'prometheus_push_gateway': None,
            'metrics_endpoints': {
                'application': 'http://localhost:8000/metrics',
                'health': 'http://localhost:8000/health'
            },
            'system_metrics': {
                'cpu': True,
                'memory': True,
                'disk': True,
                'network': True
            }
        }
        
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Сбор системных метрик"""
        logger.debug("Сбор системных метрик...")
        
        metrics = {}
        
        # CPU
        if self.config['system_metrics'].get('cpu', True):
            metrics['cpu'] = {
                'percent': psutil.cpu_percent(interval=1),
                'percent_per_cpu': psutil.cpu_percent(interval=1, percpu=True),
                'count': psutil.cpu_count(),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        
        # Память
        if self.config['system_metrics'].get('memory', True):
            memory = psutil.virtual_memory()
            metrics['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'free': memory.free,
                'percent': memory.percent
            }
        
        # Диск
        if self.config['system_metrics'].get('disk', True):
            disk = psutil.disk_usage('/')
            metrics['disk'] = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
            
            # IO статистика
            try:
                disk_io = psutil.disk_io_counters()
                metrics['disk_io'] = {
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                }
            except:
                pass
        
        # Сеть
        if self.config['system_metrics'].get('network', True):
            try:
                net_io = psutil.net_io_counters()
                metrics['network'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            except:
                pass
        
        # Информация о процессах
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            metrics['processes'] = {
                'total': len(processes),
                'top_cpu': sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:5],
                'top_memory': sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:5]
            }
        except:
            pass
        
        metrics['timestamp'] = datetime.now().isoformat()
        return metrics
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """Сбор метрик приложения"""
        logger.debug("Сбор метрик приложения...")
        
        metrics = {}
        endpoints = self.config.get('metrics_endpoints', {})
        
        for name, url in endpoints.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    try:
                        metrics[name] = response.json()
                    except:
                        metrics[name] = response.text
                else:
                    metrics[name] = {'error': f'HTTP {response.status_code}'}
            except Exception as e:
                metrics[name] = {'error': str(e)}
        
        return metrics
    
    def collect_custom_metrics(self) -> Dict[str, Any]:
        """Сбор пользовательских метрик"""
        logger.debug("Сбор пользовательских метрик...")
        
        metrics = {}

        try:
            model_metrics_path = Path('models/trained/performance_metrics.json')
            if model_metrics_path.exists():
                with open(model_metrics_path, 'r') as f:
                    model_metrics = json.load(f)
                    metrics['model'] = model_metrics
        except Exception as e:
            logger.warning(f"Не удалось загрузить метрики модели: {e}")

        try:
            drift_report_path = Path('reports/drift_monitoring/drift_report_latest.json')
            if drift_report_path.exists():
                with open(drift_report_path, 'r') as f:
                    drift_metrics = json.load(f)
                    metrics['drift'] = drift_metrics.get('summary', {})
        except:
            pass
        
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Сбор всех метрик"""
        logger.info("Сбор всех метрик...")
        
        self.metrics = {
            'system': self.collect_system_metrics(),
            'application': self.collect_application_metrics(),
            'custom': self.collect_custom_metrics(),
            'timestamp': datetime.now().isoformat(),
            'hostname': psutil.users()[0].name if psutil.users() else 'unknown'
        }
        
        return self.metrics
    
    def save_metrics(self, output_dir: str) -> Path:
        """Сохранение метрик в файл"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        latest_path = output_path / "metrics_latest.json"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(filename)
        
        logger.info(f"Метрики сохранены в: {filepath}")
        return filepath
    
    def push_to_prometheus(self) -> bool:
        """Отправка метрик в Prometheus PushGateway"""
        push_gateway = self.config.get('prometheus_push_gateway')
        
        if not push_gateway:
            logger.debug("Prometheus PushGateway не настроен, пропускаем отправку")
            return False
        
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
            
            registry = CollectorRegistry()
            
            # Системные метрики
            if 'system' in self.metrics:
                sys_metrics = self.metrics['system']
                
                if 'cpu' in sys_metrics:
                    g = Gauge('system_cpu_percent', 'CPU utilization', registry=registry)
                    g.set(sys_metrics['cpu']['percent'])
                
                if 'memory' in sys_metrics:
                    g = Gauge('system_memory_percent', 'Memory utilization', registry=registry)
                    g.set(sys_metrics['memory']['percent'])
                
                if 'disk' in sys_metrics:
                    g = Gauge('system_disk_percent', 'Disk utilization', registry=registry)
                    g.set(sys_metrics['disk']['percent'])
            
            # Отправка в PushGateway
            push_to_gateway(push_gateway, job='credit_scoring_monitoring', registry=registry)
            
            logger.info(f"Метрики отправлены в Prometheus PushGateway: {push_gateway}")
            return True
            
        except ImportError:
            logger.warning("prometheus_client не установлен, невозможно отправить метрики")
            return False
        except Exception as e:
            logger.error(f"Ошибка при отправке метрик в Prometheus: {e}")
            return False
    
    def run_continuous_collection(self):
        """Непрерывный сбор метрик"""
        logger.info("Запуск непрерывного сбора метрик...")
        logger.info(f"Интервал сбора: {self.config['collection_interval']} секунд")
        
        try:
            while True:
                start_time = time.time()
                
                # Сбор метрик
                self.collect_all_metrics()
                
                # Сохранение в файл
                output_dir = self.config.get('output_dir', 'reports/metrics')
                self.save_metrics(output_dir)
                
                # Отправка в Prometheus если настроено
                self.push_to_prometheus()
                
                # Ожидание до следующего сбора
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config['collection_interval'] - elapsed)
                
                logger.debug(f"Сбор метрик занял {elapsed:.2f} сек, ожидание {sleep_time:.2f} сек")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Остановка сбора метрик...")
        except Exception as e:
            logger.error(f"Ошибка при сборе метрик: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Сбор метрик для мониторинга')
    parser.add_argument('--config', default='configs/monitoring_config.yaml',
                       help='Файл конфигурации')
    parser.add_argument('--output-dir', default='reports/metrics',
                       help='Директория для сохранения метрик')
    parser.add_argument('--continuous', action='store_true',
                       help='Непрерывный сбор метрик')
    parser.add_argument('--interval', type=int, default=30,
                       help='Интервал сбора в секундах (для непрерывного режима)')
    parser.add_argument('--prometheus-push-gateway',
                       help='URL Prometheus PushGateway')
    
    args = parser.parse_args()
    
    # Инициализация коллектора
    collector = MetricsCollector(args.config)
    
    # Обновление конфигурации из аргументов
    if args.output_dir:
        collector.config['output_dir'] = args.output_dir
    if args.interval:
        collector.config['collection_interval'] = args.interval
    if args.prometheus_push_gateway:
        collector.config['prometheus_push_gateway'] = args.prometheus_push_gateway
    
    if args.continuous:
        # Непрерывный сбор
        collector.run_continuous_collection()
    else:
        # Единоразовый сбор
        collector.collect_all_metrics()
        filepath = collector.save_metrics(args.output_dir)
        collector.push_to_prometheus()
        
        logger.info(f"Сбор метрик завершен. Сохранено в: {filepath}")
        
        # Вывод сводки
        print("\nСводка метрик:")
        print(f"  Время сбора: {collector.metrics.get('timestamp')}")
        print(f"  Системные метрики: {'есть' if 'system' in collector.metrics else 'нет'}")
        print(f"  Метрики приложения: {'есть' if 'application' in collector.metrics else 'нет'}")
        print(f"  Пользовательские метрики: {'есть' if 'custom' in collector.metrics else 'нет'}")

if __name__ == "__main__":
    main()