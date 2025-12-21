"""
DAG для автоматического переобучения и мониторинга модели кредитного скоринга
"""
import os
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.models import Variable

# Конфигурация из Airflow Variables
SLACK_WEBHOOK_CONN_ID = Variable.get("slack_webhook_conn_id", default_var="slack_default")
K8S_NAMESPACE = Variable.get("k8s_namespace", default_var="ml-production")
MODEL_REGISTRY = Variable.get("model_registry", default_var="cr.yandex.cloud")
PROJECT_ID = Variable.get("project_id", default_var="credit-scoring")

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=2),
}

def get_env_config():
    """Получение конфигурации из переменных окружения"""
    return {
        'environment': os.getenv('ENVIRONMENT', 'staging'),
        'drift_threshold': float(Variable.get("drift_threshold", 0.3)),
        'model_version': Variable.get("model_version", "v1.0.0"),
        'data_path': Variable.get("data_path", "s3://credit-scoring-data/processed/train.csv"),
        'retraining_enabled': Variable.get("retraining_enabled", True),
    }

def check_drift(**context):
    """Проверка дрифта данных с использованием Evidently AI"""
    import requests
    import logging
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    config = get_env_config()
    
    try:
        # В реальном проекте используем Evidently или собственный сервис мониторинга
        # Здесь используем HTTP запрос к сервису мониторинга
        response = requests.get(
            f"http://drift-monitor.{K8S_NAMESPACE}.svc.cluster.local:8080/api/drift/check",
            params={
                'model_version': config['model_version'],
                'window_hours': 24,
                'threshold': config['drift_threshold']
            },
            timeout=30
        )
        
        if response.status_code == 200:
            drift_data = response.json()
            logger.info(f"Drift check completed: {drift_data}")
            
            # Push data to XCom для следующих тасков
            context['ti'].xcom_push(key='drift_report', value=drift_data)
            
            should_retrain = drift_data.get('should_retrain', False)
            reasons = drift_data.get('reasons', [])
            
            if should_retrain:
                logger.info(f"Retraining required. Reasons: {reasons}")
                return 'retrain_decision_task'
            else:
                logger.info("No retraining required")
                return 'no_retraining_needed'
        else:
            logger.error(f"Drift check failed: {response.status_code}")
            return 'drift_check_failed'
            
    except Exception as e:
        logger.error(f"Error in drift check: {str(e)}")
        return 'drift_check_failed'

def evaluate_model_quality(**context):
    """Оценка качества новой модели"""
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, roc_auc_score
    import mlflow
    
    drift_data = context['ti'].xcom_pull(key='drift_report')
    
    # Здесь должна быть логика оценки модели
    # Например, сравнение с текущей production моделью
    
    # Если качество улучшилось на 2% или больше
    quality_improved = True  # Заглушка
    
    if quality_improved:
        return 'deploy_new_model'
    else:
        return 'model_evaluation_failed'

def notify_slack(context, message, color="#36a64f"):
    """Отправка уведомления в Slack"""
    slack_msg = {
        'text': message,
        'username': 'MLOps Bot',
        'icon_emoji': ':robot_face:',
        'attachments': [{
            'color': color,
            'fields': [
                {'title': 'DAG', 'value': context.get('task_instance').dag_id, 'short': True},
                {'title': 'Task', 'value': context.get('task_instance').task_id, 'short': True},
                {'title': 'Execution Time', 'value': context.get('execution_date'), 'short': True},
            ]
        }]
    }
    
    slack_webhook = SlackWebhookOperator(
        task_id='slack_notification',
        slack_webhook_conn_id=SLACK_WEBHOOK_CONN_ID,
        message=slack_msg,
        dag=dag
    )
    
    slack_webhook.execute(context)

# Создание DAG
dag = DAG(
    dag_id='credit_scoring_retraining_pipeline',
    default_args=default_args,
    description='Automated retraining pipeline for credit scoring model',
    schedule_interval='0 2 * * *',  # Ежедневно в 2:00
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'retraining', 'credit-scoring'],
    on_success_callback=lambda context: notify_slack(
        context, 
        f"DAG {context['dag'].dag_id} completed successfully"
    ),
    on_failure_callback=lambda context: notify_slack(
        context, 
        f"DAG {context['dag'].dag_id} failed", 
        color="#ff0000"
    ),
)

# Таски
start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

check_data_drift = BranchPythonOperator(
    task_id='check_data_drift',
    python_callable=check_drift,
    provide_context=True,
    dag=dag,
)

no_retraining_needed = DummyOperator(
    task_id='no_retraining_needed',
    dag=dag,
)

drift_check_failed = DummyOperator(
    task_id='drift_check_failed',
    dag=dag,
)

retrain_decision_task = DummyOperator(
    task_id='retrain_decision_task',
    dag=dag,
)

# Переобучение модели с разными стратегиями
retrain_model_simple = KubernetesPodOperator(
    task_id='retrain_model_simple',
    namespace=K8S_NAMESPACE,
    image=f"{MODEL_REGISTRY}/{PROJECT_ID}-training:latest",
    cmds=["python", "/app/scripts/retrain.py"],
    arguments=[
        "--data-path", "s3://credit-scoring-data/processed/train.csv",
        "--model-type", "random_forest",
        "--output-path", "/tmp/new_model.pkl",
        "--config", "/app/configs/model_training.yaml"
    ],
    name="retrain-simple-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    log_events_on_failure=True,
    dag=dag,
)

retrain_model_nn = KubernetesPodOperator(
    task_id='retrain_model_nn',
    namespace=K8S_NAMESPACE,
    image=f"{MODEL_REGISTRY}/{PROJECT_ID}-training:latest",
    cmds=["python", "/app/scripts/retrain_nn.py"],
    arguments=[
        "--data-path", "s3://credit-scoring-data/processed/train.csv",
        "--model-type", "neural_network",
        "--output-path", "/tmp/new_model_nn.pth",
        "--epochs", "50"
    ],
    name="retrain-nn-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

# Конвертация в ONNX
convert_to_onnx = KubernetesPodOperator(
    task_id='convert_to_onnx',
    namespace=K8S_NAMESPACE,
    image=f"{MODEL_REGISTRY}/{PROJECT_ID}-training:latest",
    cmds=["python", "/app/scripts/convert_to_onnx.py"],
    arguments=[
        "--input-model", "/tmp/new_model.pkl",
        "--output-model", "/tmp/new_model.onnx",
        "--optimize"
    ],
    name="convert-onnx-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

# Оптимизация модели (Quantization)
optimize_model = KubernetesPodOperator(
    task_id='optimize_model',
    namespace=K8S_NAMESPACE,
    image=f"{MODEL_REGISTRY}/{PROJECT_ID}-training:latest",
    cmds=["python", "/app/scripts/optimize_model.py"],
    arguments=[
        "--model", "/tmp/new_model.onnx",
        "--output", "/tmp/new_model_quantized.onnx",
        "--quantize",
        "--precision", "int8"
    ],
    name="optimize-model-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

# Валидация модели
validate_model = BranchPythonOperator(
    task_id='validate_model',
    python_callable=evaluate_model_quality,
    provide_context=True,
    dag=dag,
)

model_evaluation_failed = DummyOperator(
    task_id='model_evaluation_failed',
    dag=dag,
)

# Тестирование нагрузки
load_test_model = KubernetesPodOperator(
    task_id='load_test_model',
    namespace=K8S_NAMESPACE,
    image=f"{MODEL_REGISTRY}/{PROJECT_ID}-testing:latest",
    cmds=["python", "/app/scripts/load_test.py"],
    arguments=[
        "--model-path", "/tmp/new_model_quantized.onnx",
        "--requests-per-second", "100",
        "--duration", "60"
    ],
    name="load-test-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

# Развертывание с blue-green стратегией
deploy_new_model = KubernetesPodOperator(
    task_id='deploy_new_model',
    namespace=K8S_NAMESPACE,
    image=f"{MODEL_REGISTRY}/{PROJECT_ID}-deployer:latest",
    cmds=["python", "/app/scripts/deploy_blue_green.py"],
    arguments=[
        "--model-path", "/tmp/new_model_quantized.onnx",
        "--environment", "staging",
        "--strategy", "blue-green",
        "--validation-timeout", "300"
    ],
    name="deploy-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

# Канареечное развертывание (опционально)
deploy_canary = KubernetesPodOperator(
    task_id='deploy_canary',
    namespace=K8S_NAMESPACE,
    image=f"{MODEL_REGISTRY}/{PROJECT_ID}-deployer:latest",
    cmds=["python", "/app/scripts/deploy_canary.py"],
    arguments=[
        "--model-path", "/tmp/new_model_quantized.onnx",
        "--environment", "production",
        "--strategy", "canary",
        "--steps", "10,25,50,100",
        "--step-duration", "300"
    ],
    name="deploy-canary-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

# Обновление мониторинга
update_monitoring = KubernetesPodOperator(
    task_id='update_monitoring',
    namespace=K8S_NAMESPACE,
    image=f"{MODEL_REGISTRY}/{PROJECT_ID}-monitoring:latest",
    cmds=["python", "/app/scripts/update_monitoring.py"],
    arguments=[
        "--model-version", Variable.get("model_version", "v1.0.0"),
        "--reference-data", "s3://credit-scoring-data/processed/train.csv"
    ],
    name="update-monitoring-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

end_pipeline = DummyOperator(
    task_id='end_pipeline',
    trigger_rule='none_failed_or_skipped',
    dag=dag,
)

# Обработка сбоев
handle_failure = SlackWebhookOperator(
    task_id='handle_failure',
    slack_webhook_conn_id=SLACK_WEBHOOK_CONN_ID,
    message="❌ Model retraining pipeline failed",
    trigger_rule='one_failed',
    dag=dag,
)

# Оркестрация
start_pipeline >> check_data_drift

check_data_drift >> [no_retraining_needed, drift_check_failed, retrain_decision_task]

retrain_decision_task >> retrain_model_simple
retrain_model_simple >> convert_to_onnx >> optimize_model >> validate_model

validate_model >> [model_evaluation_failed, load_test_model]
load_test_model >> deploy_new_model >> update_monitoring

# Параллельное переобучение нейронной сети (опционально)
retrain_decision_task >> retrain_model_nn
retrain_model_nn >> convert_to_onnx

# Соединение концов
[no_retraining_needed, drift_check_failed, model_evaluation_failed, update_monitoring] >> end_pipeline

# Обработка ошибок
[retrain_model_simple, retrain_model_nn, convert_to_onnx, optimize_model, 
 validate_model, load_test_model, deploy_new_model, update_monitoring] >> handle_failure