"""
Проверка здоровья сервисов
Утилита для проверки доступности и здоровья всех компонентов системы
"""
import requests
import time
import json
import socket
import psutil
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import Dict, List, Tuple, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthChecker:
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.results = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        default_config = {
            'services': {
                'api': {
                    'url': 'http://localhost:8000',
                    'endpoints': ['/health/live', '/health/ready', '/metrics'],
                    'timeout': 5
                },
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'type': 'postgresql'
                },
                'redis': {
                    'host': 'localhost',
                    'port': 6379
                },
                'prometheus': {
                    'url': 'http://localhost:9090'
                },
                'grafana': {
                    'url': 'http://localhost:3000'
                }
            },
            'system_checks': {
                'cpu_threshold': 90,
                'memory_threshold': 90,
                'disk_threshold': 90
            }
        }
        
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config
    
    def check_http_service(self, name: str, url: str, endpoint: str = '/', timeout: int = 5) -> Dict:
        """Проверка HTTP сервиса"""
        full_url = f"{url.rstrip('/')}{endpoint}"
        
        try:
            start_time = time.time()
            response = requests.get(full_url, timeout=timeout)
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy' if response.status_code < 500 else 'unhealthy',
                'http_code': response.status_code,
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.Timeout:
            return {
                'status': 'timeout',
                'error': 'Request timeout',
                'response_time_seconds': timeout,
                'timestamp': datetime.now().isoformat()
            }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'unreachable',
                'error': 'Connection refused',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_port(self, host: str, port: int, timeout: int = 5) -> Dict:
        """Проверка доступности порта"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            start_time = time.time()
            result = sock.connect_ex((host, port))
            response_time = time.time() - start_time
            
            sock.close()
            
            if result == 0:
                return {
                    'status': 'healthy',
                    'response_time_seconds': response_time,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'unreachable',
                    'error': f'Port {port} is closed',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_database(self, name: str, config: Dict) -> Dict:
        """Проверка базы данных"""
        db_type = config.get('type', 'postgresql')
        
        if db_type == 'postgresql':
            try:
                import psycopg2
                
                host = config.get('host', 'localhost')
                port = config.get('port', 5432)
                database = config.get('database', 'postgres')
                user = config.get('user', 'postgres')
                password = config.get('password', '')
                
                start_time = time.time()
                
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                    connect_timeout=5
                )
                
                response_time = time.time() - start_time
                
                # Проверяем возможность выполнения запроса
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                
                cursor.close()
                conn.close()
                
                return {
                    'status': 'healthy',
                    'response_time_seconds': response_time,
                    'timestamp': datetime.now().isoformat()
                }
                
            except ImportError:
                return {
                    'status': 'error',
                    'error': 'psycopg2 not installed',
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        elif db_type == 'redis':
            host = config.get('host', 'localhost')
            port = config.get('port', 6379)
            
            # Проверяем порт
            port_check = self.check_port(host, port)
            
            if port_check['status'] == 'healthy':
                try:
                    import redis
                    
                    start_time = time.time()
                    
                    r = redis.Redis(
                        host=host,
                        port=port,
                        socket_connect_timeout=5
                    )
                    
                    # Пинг Redis
                    r.ping()
                    response_time = time.time() - start_time
                    
                    return {
                        'status': 'healthy',
                        'response_time_seconds': response_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                except ImportError:
                    return {
                        **port_check,
                        'warning': 'redis-py not installed'
                    }
                except Exception as e:
                    return {
                        'status': 'unhealthy',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return port_check
        
        else:
            return {
                'status': 'unknown',
                'error': f'Unsupported database type: {db_type}',
                'timestamp': datetime.now().isoformat()
            }
    
    def check_system_resources(self) -> Dict:
        """Проверка системных ресурсов"""
        metrics = {}
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics['cpu'] = {
            'percent': cpu_percent,
            'status': 'healthy' if cpu_percent < self.config['system_checks']['cpu_threshold'] else 'warning',
            'threshold': self.config['system_checks']['cpu_threshold']
        }
        
        # Память
        memory = psutil.virtual_memory()
        metrics['memory'] = {
            'percent': memory.percent,
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'status': 'healthy' if memory.percent < self.config['system_checks']['memory_threshold'] else 'warning',
            'threshold': self.config['system_checks']['memory_threshold']
        }
        
        # Диск
        disk = psutil.disk_usage('/')
        metrics['disk'] = {
            'percent': disk.percent,
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'status': 'healthy' if disk.percent < self.config['system_checks']['disk_threshold'] else 'warning',
            'threshold': self.config['system_checks']['disk_threshold']
        }
        
        # Загрузка системы
        try:
            load_avg = psutil.getloadavg()
            metrics['load_average'] = {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2],
                'status': 'healthy' if load_avg[0] < psutil.cpu_count() else 'warning'
            }
        except:
            pass
        
        metrics['timestamp'] = datetime.now().isoformat()
        return metrics
    
    def check_kubernetes(self, namespace: str = None) -> Dict:
        """Проверка состояния Kubernetes"""
        try:
            import subprocess
            
            cmd = ['kubectl', 'get', 'pods']
            if namespace:
                cmd.extend(['-n', namespace])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return {
                    'status': 'error',
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Парсим вывод
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:
                return {
                    'status': 'healthy',
                    'pods': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            pods = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    pods.append({
                        'name': parts[0],
                        'ready': parts[1],
                        'status': parts[2]
                    })
            
            # Анализируем статусы подов
            unhealthy_pods = [p for p in pods if p['status'] not in ['Running', 'Completed']]
            
            return {
                'status': 'healthy' if len(unhealthy_pods) == 0 else 'warning',
                'total_pods': len(pods),
                'unhealthy_pods': len(unhealthy_pods),
                'pods': pods,
                'timestamp': datetime.now().isoformat()
            }
            
        except ImportError:
            return {
                'status': 'error',
                'error': 'kubectl not available',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_model_service(self) -> Dict:
        """Проверка службы модели"""
        # Проверяем наличие модели
        model_path = Path('models/trained/credit_scoring_model.pkl')
        
        if not model_path.exists():
            return {
                'status': 'unhealthy',
                'error': 'Model file not found',
                'timestamp': datetime.now().isoformat()
            }
        
        # Проверяем метрики модели
        metrics_path = Path('models/trained/performance_metrics.json')
        
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                model_health = {
                    'status': 'healthy',
                    'model_exists': True,
                    'metrics_available': True,
                    'last_trained': metrics.get('model_info', {}).get('training_date', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Добавляем ключевые метрики
                if 'test' in metrics:
                    for metric in ['accuracy', 'roc_auc', 'f1_score']:
                        if metric in metrics['test']:
                            model_health[metric] = metrics['test'][metric]
                
                return model_health
                
            except Exception as e:
                return {
                    'status': 'warning',
                    'model_exists': True,
                    'metrics_available': False,
                    'error': f'Cannot read metrics: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
        else:
            return {
                'status': 'warning',
                'model_exists': True,
                'metrics_available': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_checks(self) -> Dict:
        """Запуск всех проверок"""
        logger.info("Запуск комплексной проверки здоровья...")
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'system': {},
            'overall_status': 'healthy'
        }
        
        # Проверка сервисов из конфигурации
        services = self.config.get('services', {})
        
        for service_name, service_config in services.items():
            logger.info(f"Проверка сервиса: {service_name}")
            
            if 'url' in service_config:
                # HTTP сервис
                endpoints = service_config.get('endpoints', ['/'])
                checks = {}
                
                for endpoint in endpoints:
                    check_result = self.check_http_service(
                        service_name,
                        service_config['url'],
                        endpoint,
                        service_config.get('timeout', 5)
                    )
                    checks[endpoint] = check_result
                
                self.results['services'][service_name] = {
                    'type': 'http',
                    'checks': checks,
                    'overall_status': 'healthy' if all(
                        c['status'] == 'healthy' for c in checks.values()
                    ) else 'unhealthy'
                }
                
            elif 'host' in service_config and 'port' in service_config:
                # Сервис с портом (база данных, кэш и т.д.)
                if service_config.get('type') in ['postgresql', 'redis']:
                    check_result = self.check_database(service_name, service_config)
                else:
                    check_result = self.check_port(
                        service_config['host'],
                        service_config['port'],
                        service_config.get('timeout', 5)
                    )
                
                self.results['services'][service_name] = {
                    'type': service_config.get('type', 'tcp'),
                    'check': check_result,
                    'overall_status': check_result['status']
                }
        
        # Проверка системных ресурсов
        logger.info("Проверка системных ресурсов...")
        self.results['system'] = self.check_system_resources()
        
        # Проверка Kubernetes (если доступно)
        logger.info("Проверка Kubernetes...")
        k8s_namespace = self.config.get('kubernetes', {}).get('namespace')
        self.results['kubernetes'] = self.check_kubernetes(k8s_namespace)
        
        # Проверка службы модели
        logger.info("Проверка службы модели...")
        self.results['model_service'] = self.check_model_service()
        
        # Определение общего статуса
        self.determine_overall_status()
        
        return self.results
    
    def determine_overall_status(self):
        """Определение общего статуса системы"""
        all_statuses = []
        
        # Собираем статусы сервисов
        for service_name, service_result in self.results.get('services', {}).items():
            all_statuses.append(service_result.get('overall_status', 'unknown'))
        
        # Добавляем статусы системных проверок
        for component in ['system', 'kubernetes', 'model_service']:
            if component in self.results:
                status = self.results[component].get('status', 'unknown')
                all_statuses.append(status)
        
        # Определяем общий статус
        if 'unhealthy' in all_statuses:
            self.results['overall_status'] = 'unhealthy'
        elif 'warning' in all_statuses:
            self.results['overall_status'] = 'warning'
        elif 'error' in all_statuses:
            self.results['overall_status'] = 'error'
        elif all(status == 'healthy' for status in all_statuses):
            self.results['overall_status'] = 'healthy'
        else:
            self.results['overall_status'] = 'unknown'
    
    def generate_report(self, output_dir: str = 'reports/health_checks') -> Tuple[Path, Path]:
        """Генерация отчета о проверке здоровья"""
        logger.info("Генерация отчета...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение JSON отчета
        json_report_path = output_path / f"health_check_{timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Генерация HTML отчета
        html_report_path = output_path / f"health_check_{timestamp}.html"
        self.generate_html_report(html_report_path)
        
        # Создание симлинка на последний отчет
        latest_path = output_path / "health_check_latest.json"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(json_report_path.name)
        
        logger.info(f"Отчеты сохранены в: {output_dir}")
        return json_report_path, html_report_path
    
    def generate_html_report(self, output_path: Path):
        """Генерация HTML отчета"""
        status_colors = {
            'healthy': '#4CAF50',
            'warning': '#FF9800',
            'unhealthy': '#F44336',
            'error': '#D32F2F',
            'unknown': '#9E9E9E'
        }
        
        overall_status = self.results.get('overall_status', 'unknown')
        overall_color = status_colors.get(overall_status, '#9E9E9E')
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Отчет проверки здоровья - {self.results.get('timestamp')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .overall-status {{ 
                    display: inline-block; 
                    padding: 10px 20px; 
                    background: {overall_color}; 
                    color: white; 
                    border-radius: 5px; 
                    font-weight: bold; 
                    font-size: 1.2em;
                }}
                .section {{ 
                    margin: 20px 0; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                }}
                .service {{ 
                    margin: 10px 0; 
                    padding: 10px; 
                    border-left: 4px solid #ddd; 
                }}
                .status-indicator {{
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 5px;
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 10px 0; 
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #f2f2f2; 
                }}
                .healthy {{ border-left-color: #4CAF50; }}
                .warning {{ border-left-color: #FF9800; }}
                .unhealthy {{ border-left-color: #F44336; }}
                .error {{ border-left-color: #D32F2F; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Отчет проверки здоровья системы</h1>
                <p>Время проверки: {self.results.get('timestamp')}</p>
                <div class="overall-status">
                    Общий статус: {overall_status.upper()}
                </div>
            </div>
        """
        
        # Секция сервисов
        html_content += '<div class="section"><h2>Сервисы</h2>'
        
        for service_name, service_result in self.results.get('services', {}).items():
            status = service_result.get('overall_status', 'unknown')
            color = status_colors.get(status, '#9E9E9E')
            
            html_content += f"""
            <div class="service {status}">
                <h3>
                    <span class="status-indicator" style="background-color: {color};"></span>
                    {service_name} ({service_result.get('type', 'unknown')})
                </h3>
            """
            
            if 'checks' in service_result:
                html_content += '<table><tr><th>Endpoint</th><th>Статус</th><th>HTTP код</th><th>Время ответа</th></tr>'
                
                for endpoint, check in service_result['checks'].items():
                    endpoint_color = status_colors.get(check.get('status', 'unknown'), '#9E9E9E')
                    http_code = check.get('http_code', 'N/A')
                    response_time = f"{check.get('response_time_seconds', 0):.3f}s" if 'response_time_seconds' in check else 'N/A'
                    
                    html_content += f"""
                    <tr>
                        <td>{endpoint}</td>
                        <td style="color: {endpoint_color}">{check.get('status', 'unknown')}</td>
                        <td>{http_code}</td>
                        <td>{response_time}</td>
                    </tr>
                    """
                
                html_content += '</table>'
            elif 'check' in service_result:
                check = service_result['check']
                html_content += f"<p>Статус: {check.get('status', 'unknown')}</p>"
                if 'error' in check:
                    html_content += f"<p>Ошибка: {check.get('error')}</p>"
            
            html_content += '</div>'
        
        html_content += '</div>'
        
        # Секция системных ресурсов
        if 'system' in self.results:
            sys_result = self.results['system']
            html_content += '<div class="section"><h2>Системные ресурсы</h2><table>'
            
            for component, data in sys_result.items():
                if component == 'timestamp':
                    continue
                
                if isinstance(data, dict):
                    status = data.get('status', 'unknown')
                    color = status_colors.get(status, '#9E9E9E')
                    
                    html_content += f'<tr><td>{component}</td><td style="color: {color}">{status}</td>'
                    
                    if 'percent' in data:
                        html_content += f'<td>{data["percent"]:.1f}%</td>'
                    else:
                        html_content += '<td>N/A</td>'
                    
                    html_content += '</tr>'
            
            html_content += '</table></div>'
        
        # Секция Kubernetes
        if 'kubernetes' in self.results:
            k8s_result = self.results['kubernetes']
            status = k8s_result.get('status', 'unknown')
            color = status_colors.get(status, '#9E9E9E')
            
            html_content += f"""
            <div class="section">
                <h2>Kubernetes</h2>
                <p>Статус: <span style="color: {color}">{status}</span></p>
                <p>Всего подов: {k8s_result.get('total_pods', 0)}</p>
                <p>Нездоровых подов: {k8s_result.get('unhealthy_pods', 0)}</p>
            </div>
            """
        
        # Секция модели
        if 'model_service' in self.results:
            model_result = self.results['model_service']
            status = model_result.get('status', 'unknown')
            color = status_colors.get(status, '#9E9E9E')
            
            html_content += f"""
            <div class="section">
                <h2>Служба модели</h2>
                <p>Статус: <span style="color: {color}">{status}</span></p>
                <p>Модель существует: {'Да' if model_result.get('model_exists') else 'Нет'}</p>
                <p>Метрики доступны: {'Да' if model_result.get('metrics_available') else 'Нет'}</p>
            """
            
            if 'accuracy' in model_result:
                html_content += f'<p>Accuracy: {model_result["accuracy"]:.4f}</p>'
            if 'roc_auc' in model_result:
                html_content += f'<p>ROC-AUC: {model_result["roc_auc"]:.4f}</p>'
            
            html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def print_summary(self):
        """Вывод сводки в консоль"""
        print("\n" + "=" * 60)
        print("СВОДКА ПРОВЕРКИ ЗДОРОВЬЯ")
        print("=" * 60)
        
        print(f"\nОбщий статус: {self.results.get('overall_status', 'unknown').upper()}")
        print(f"Время проверки: {self.results.get('timestamp')}")
        
        # Сервисы
        print("\nСервисы:")
        for service_name, service_result in self.results.get('services', {}).items():
            status = service_result.get('overall_status', 'unknown')
            status_icon = "✅" if status == 'healthy' else "⚠️" if status == 'warning' else "❌"
            print(f"  {status_icon} {service_name}: {status}")
        
        # Системные ресурсы
        if 'system' in self.results:
            sys_result = self.results['system']
            print("\nСистемные ресурсы:")
            
            for component, data in sys_result.items():
                if component == 'timestamp':
                    continue
                
                if isinstance(data, dict) and 'percent' in data:
                    status = data.get('status', 'unknown')
                    status_icon = "✅" if status == 'healthy' else "⚠️"
                    print(f"  {status_icon} {component}: {data['percent']:.1f}% ({status})")
        
        # Рекомендации
        print("\n" + "=" * 60)
        print("РЕКОМЕНДАЦИИ")
        print("=" * 60)
        
        if self.results.get('overall_status') == 'healthy':
            print("✅ Все системы работают нормально")
        else:
            print("⚠️  Обнаружены проблемы:")
            
            for service_name, service_result in self.results.get('services', {}).items():
                if service_result.get('overall_status') not in ['healthy', 'unknown']:
                    print(f"  • Сервис '{service_name}' имеет статус: {service_result['overall_status']}")
            
            if 'system' in self.results:
                for component, data in self.results['system'].items():
                    if isinstance(data, dict) and data.get('status') == 'warning':
                        print(f"  • {component}: {data.get('percent', 0):.1f}% (превышен порог)")

def main():
    parser = argparse.ArgumentParser(description='Проверка здоровья сервисов')
    parser.add_argument('--config', default='configs/health_check_config.yaml',
                       help='Файл конфигурации')
    parser.add_argument('--output-dir', default='reports/health_checks',
                       help='Директория для сохранения отчетов')
    parser.add_argument('--continuous', action='store_true',
                       help='Непрерывная проверка')
    parser.add_argument('--interval', type=int, default=60,
                       help='Интервал проверки в секундах')
    
    args = parser.parse_args()
    
    # Инициализация проверяльщика
    checker = HealthChecker(args.config)
    
    if args.continuous:
        # Непрерывная проверка
        logger.info(f"Запуск непрерывной проверки с интервалом {args.interval} секунд")
        
        try:
            while True:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Запуск проверки...")
                
                checker.run_all_checks()
                checker.print_summary()
                
                # Сохранение отчета
                checker.generate_report(args.output_dir)
                
                print(f"\nОжидание {args.interval} секунд до следующей проверки...")
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\nОстановка непрерывной проверки...")
    else:
        # Одноразовая проверка
        checker.run_all_checks()
        checker.print_summary()
        
        # Сохранение отчета
        json_report, html_report = checker.generate_report(args.output_dir)
        
        print(f"\nОтчеты сохранены:")
        print(f"  JSON: {json_report}")
        print(f"  HTML: {html_report}")

if __name__ == "__main__":
    main()