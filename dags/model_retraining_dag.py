from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'credit_scoring_retraining',
    default_args=default_args,
    description='DAG for automatic model retraining',
    schedule_interval='0 2 * * *',  # Ежедневно в 2:00
    catchup=False,
    tags=['mlops', 'retraining']
)

def check_drift(**context):
    """Проверка дрифта данных"""
    import requests
    import json
    
    # Вызов мониторинга дрифта
    response = requests.get('http://drift-monitor:8000/api/monitoring/drift')
    drift_data = response.json()
    
    # Передача данных в следующий таск
    context['ti'].xcom_push(key='drift_data', value=drift_data)
    
    return drift_data['should_retrain']

def trigger_retraining(**context):
    """Запуск переобучения при обнаружении дрифта"""
    drift_data = context['ti'].xcom_pull(key='drift_data')
    
    if drift_data.get('should_retrain'):
        # Логирование причины
        print(f"Запуск переобучения. Причина: {drift_data.get('reasons', [])}")
        return True
    return False

check_drift_task = PythonOperator(
    task_id='check_data_drift',
    python_callable=check_drift,
    provide_context=True,
    dag=dag,
)

retrain_decision_task = PythonOperator(
    task_id='decide_retraining',
    python_callable=trigger_retraining,
    provide_context=True,
    dag=dag,
)

retrain_model_task = KubernetesPodOperator(
    task_id='retrain_model',
    namespace='ml-production',
    image='registry.yandexcloud.net/mlops/credit-scoring-training:latest',
    cmds=["python", "/app/retrain.py"],
    arguments=["--trigger", "scheduled"],
    name="retrain-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

validate_model_task = KubernetesPodOperator(
    task_id='validate_model',
    namespace='ml-production',
    image='registry.yandexcloud.net/mlops/credit-scoring-training:latest',
    cmds=["python", "/app/validate_model.py"],
    name="validate-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

deploy_model_task = KubernetesPodOperator(
    task_id='deploy_new_model',
    namespace='ml-production',
    image='registry.yandexcloud.net/mlops/deployer:latest',
    cmds=["python", "/app/deploy.py"],
    arguments=["--strategy", "canary"],
    name="deploy-pod",
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

slack_notification = SlackWebhookOperator(
    task_id='slack_notification',
    slack_webhook_conn_id='slack_webhook',
    message='✅ Модель успешно переобучена и развернута',
    dag=dag,
)

# Оркестрация пайплайна
check_drift_task >> retrain_decision_task
retrain_decision_task >> retrain_model_task
retrain_model_task >> validate_model_task >> deploy_model_task >> slack_notification