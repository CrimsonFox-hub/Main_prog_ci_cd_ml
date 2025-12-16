"""
Airflow DAG для автоматического переобучения модели
Этап 7: Пайплайн переобучения и автоматизация
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import Variable, Connection
from airflow.utils.trigger_rule import TriggerRule
from kubernetes.client import models as k8s_models
import pendulum
import json
import yaml
from pathlib import Path

# Локальный часовой пояс
local_tz = pendulum.timezone("Europe/Moscow")

# Конфигурация по умолчанию
default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1, tzinfo=local_tz),
    'email': ['ml-team@your-bank.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
    'catchup': False,
    'execution_timeout': timedelta(hours=6)
}

# Загрузка конфигурации
with open('/opt/airflow/configs/ml_pipeline_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def check_drift(**context):
    """Проверка дрифта данных и концепта"""
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Получение результатов мониторинга
    monitoring_api = Variable.get("monitoring_api_endpoint")
    
    try:
        response = requests.get(
            f"{monitoring_api}/api/v1/drift/latest",
            headers={'Authorization': f"Bearer {Variable.get('monitoring_api_token')}"},
            timeout=30
        )
        
        if response.status_code == 200:
            drift_data = response.json()
            
            # Проверка условий дрифта
            data_drift_detected = drift_data.get('data_drift_detected', False)
            concept_drift_detected = drift_data.get('concept_drift_detected', False)
            performance_decay = drift_data.get('significant_performance_decay', False)
            
            # Проверка временных условий (последние данные)
            last_update = datetime.fromisoformat(drift_data.get('timestamp', '2000-01-01'))
            is_recent = (datetime.now() - last_update) < timedelta(days=2)
            
            should_retrain = (
                (data_drift_detected and drift_data.get('data_drift_score', 0) > 0.3) or
                (concept_drift_detected and drift_data.get('concept_drift_score', 0) > 0.25) or
                performance_decay or
                (not is_recent)  # Переобучение если данные устарели
            )
            
            # Push в XCom
            context['ti'].xcom_push(key='should_retrain', value=should_retrain)
            context['ti'].xcom_push(key='drift_data', value=drift_data)
            
            return {
                'should_retrain': should_retrain,
                'data_drift_score': drift_data.get('data_drift_score', 0),
                'concept_drift_score': drift_data.get('concept_drift_score', 0),
                'reason': 'drift_detected' if should_retrain else 'no_drift'
            }
            
    except Exception as e:
        # Если API недоступен, все равно запускаем переобучение по расписанию
        print(f"Monitoring API error: {str(e)}. Proceeding with scheduled retraining.")
        
        context['ti'].xcom_push(key='should_retrain', value=True)
        context['ti'].xcom_push(key='drift_data', value={'error': str(e)})
        
        return {
            'should_retrain': True,
            'reason': 'monitoring_unavailable'
        }

def check_data_availability(**context):
    """Проверка доступности новых данных"""
    import boto3
    from datetime import datetime, timedelta
    
    s3_client = boto3.client(
        's3',
        endpoint_url=Variable.get('s3_endpoint'),
        aws_access_key_id=Variable.get('s3_access_key'),
        aws_secret_access_key=Variable.get('s3_secret_key')
    )
    
    # Проверка наличия новых данных
    bucket_name = Variable.get('data_bucket_name')
    prefix = 'raw/credit_data/'
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=10
        )
        
        # Проверка свежести данных
        latest_data = None
        if 'Contents' in response:
            latest_data = max(response['Contents'], key=lambda x: x['LastModified'])
            data_age = datetime.now(latest_data['LastModified'].tzinfo) - latest_data['LastModified']
            
            has_new_data = data_age < timedelta(days=7)
            
            context['ti'].xcom_push(key='has_new_data', value=has_new_data)
            context['ti'].xcom_push(key='latest_data_age_days', value=data_age.days)
            
            return {
                'has_new_data': has_new_data,
                'latest_data_age_days': data_age.days,
                'latest_data_key': latest_data['Key']
            }
        else:
            context['ti'].xcom_push(key='has_new_data', value=False)
            return {'has_new_data': False, 'error': 'no_data_found'}
            
    except Exception as e:
        print(f"Error checking data availability: {str(e)}")
        context['ti'].xcom_push(key='has_new_data', value=False)
        return {'has_new_data': False, 'error': str(e)}

def decide_retraining(**context):
    """Принятие решения о переобучении"""
    should_retrain = context['ti'].xcom_pull(task_ids='check_drift', key='should_retrain')
    has_new_data = context['ti'].xcom_pull(task_ids='check_data_availability', key='has_new_data')
    
    # Получение причины из check_drift
    drift_result = context['ti'].xcom_pull(task_ids='check_drift')
    reason = drift_result.get('reason', 'scheduled') if isinstance(drift_result, dict) else 'scheduled'
    
    # Условия переобучения
    conditions_met = should_retrain or has_new_data
    
    if not conditions_met:
        print("Retraining conditions not met. Skipping...")
        context['ti'].xcom_push(key='retrain_decision', value=False)
        return {'decision': 'skip', 'reason': 'conditions_not_met'}
    
    # Определение приоритета
    priority = 'high' if should_retrain else 'medium'
    
    # Push решения в XCom
    context['ti'].xcom_push(key='retrain_decision', value=True)
    context['ti'].xcom_push(key='retrain_priority', value=priority)
    context['ti'].xcom_push(key='retrain_reason', value=reason)
    
    return {
        'decision': 'proceed',
        'priority': priority,
        'reason': reason,
        'conditions': {
            'should_retrain': should_retrain,
            'has_new_data': has_new_data
        }
    }

def send_slack_notification(**context):
    """Отправка уведомления в Slack"""
    decision = context['ti'].xcom_pull(task_ids='decide_retraining')
    
    if decision.get('decision') == 'skip':
        message = "🟡 *Scheduled Model Retraining Skipped*\n"
        message += f"Reason: {decision.get('reason', 'Conditions not met')}\n"
        color = "warning"
    else:
        message = "🟢 *Model Retraining Started*\n"
        message += f"Priority: {decision.get('priority', 'medium')}\n"
        message += f"Reason: {decision.get('reason', 'scheduled')}\n"
        message += f"DAG Run: {context['run_id']}\n"
        color = "good"
    
    slack_webhook_token = Variable.get("slack_webhook_token")
    
    slack_task = SlackWebhookOperator(
        task_id='send_slack_notification',
        http_conn_id='slack_webhook',
        message=message,
        username='Airflow ML Pipeline',
        icon_emoji=':robot_face:',
        dag=dag
    )
    
    return slack_task.execute(context)

# Секреты Kubernetes
secrets = [
    Secret(
        deploy_type='env',
        deploy_target='MLFLOW_TRACKING_URI',
        secret='mlflow-secrets',
        key='tracking-uri'
    ),
    Secret(
        deploy_type='env',
        deploy_target='DVC_REMOTE_URL',
        secret='dvc-secrets',
        key='remote-url'
    ),
    Secret(
        deploy_type='env',
        deploy_target='S3_ACCESS_KEY',
        secret='s3-secrets',
        key='access-key'
    ),
    Secret(
        deploy_type='env',
        deploy_target='S3_SECRET_KEY',
        secret='s3-secrets',
        key='secret-key'
    )
]

# Конфигурация ресурсов для тренировки
resources = k8s_models.V1ResourceRequirements(
    requests={
        'memory': '8Gi',
        'cpu': '4',
        'nvidia.com/gpu': '1' if config['training']['use_gpu'] else None
    },
    limits={
        'memory': '16Gi',
        'cpu': '8',
        'nvidia.com/gpu': '1' if config['training']['use_gpu'] else None
    }
)

# Создание DAG
with DAG(
    dag_id='credit_scoring_retraining',
    default_args=default_args,
    description='Automated retraining pipeline for credit scoring model',
    schedule_interval=timedelta(days=7),  # Еженедельно
    catchup=False,
    tags=['mlops', 'retraining', 'production'],
    concurrency=1,
    max_active_runs=1,
    on_success_callback=None,
    on_failure_callback=None,
) as dag:
    
    # Стартовая задача
    start = DummyOperator(
        task_id='start',
        dag=dag
    )
    
    # Проверка дрифта
    check_drift_task = PythonOperator(
        task_id='check_drift',
        python_callable=check_drift,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
        dag=dag
    )
    
    # Проверка доступности данных
    check_data_task = PythonOperator(
        task_id='check_data_availability',
        python_callable=check_data_availability,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        dag=dag
    )
    
    # Принятие решения
    decide_task = PythonOperator(
        task_id='decide_retraining',
        python_callable=decide_retraining,
        provide_context=True,
        dag=dag
    )
    
    # Уведомление в Slack
    notify_start = PythonOperator(
        task_id='notify_start',
        python_callable=send_slack_notification,
        provide_context=True,
        dag=dag
    )
    
    # Параллельная подготовка данных
    prepare_data = KubernetesPodOperator(
        task_id='prepare_training_data',
        namespace='ml-training',
        image=f"{Variable.get('image_registry')}/data-preparation:latest",
        cmds=['python', '-m', 'src.data_preparation.pipeline'],
        arguments=[
            '--input-path', '/data/raw',
            '--output-path', '/data/processed',
            '--config', '/app/configs/training_config.yaml'
        ],
        secrets=secrets,
        name='prepare-data-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        resources=resources,
        image_pull_policy='Always',
        env_vars={
            'EXECUTION_DATE': '{{ ds }}',
            'DAG_RUN_ID': '{{ run_id }}'
        },
        volumes=[
            k8s_models.V1Volume(
                name='data-volume',
                persistent_volume_claim=k8s_models.V1PersistentVolumeClaimVolumeSource(
                    claim_name='data-pvc'
                )
            )
        ],
        volume_mounts=[
            k8s_models.V1VolumeMount(
                name='data-volume',
                mount_path='/data'
            )
        ],
        dag=dag
    )
    
    # Обучение модели
    train_model = KubernetesPodOperator(
        task_id='train_model',
        namespace='ml-training',
        image=f"{Variable.get('image_registry')}/model-training:latest",
        cmds=['python', '-m', 'src.ml_pipeline.training.train_model'],
        arguments=[
            '--config', '/app/configs/training_config.yaml',
            '--data-path', '/data/processed/train.csv',
            '--output-path', '/models',
            '--experiment-name', 'credit_scoring_retraining_{{ ds_nodash }}'
        ],
        secrets=secrets,
        name='train-model-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        resources=resources,
        image_pull_policy='Always',
        env_vars={
            'MLFLOW_EXPERIMENT_NAME': 'credit_scoring_retraining',
            'MLFLOW_RUN_NAME': 'run_{{ ds_nodash }}_{{ ts_nodash }}'
        },
        volumes=[
            k8s_models.V1Volume(
                name='data-volume',
                persistent_volume_claim=k8s_models.V1PersistentVolumeClaimVolumeSource(
                    claim_name='data-pvc'
                )
            ),
            k8s_models.V1Volume(
                name='models-volume',
                persistent_volume_claim=k8s_models.V1PersistentVolumeClaimVolumeSource(
                    claim_name='models-pvc'
                )
            )
        ],
        volume_mounts=[
            k8s_models.V1VolumeMount(
                name='data-volume',
                mount_path='/data'
            ),
            k8s_models.V1VolumeMount(
                name='models-volume',
                mount_path='/models'
            )
        ],
        dag=dag
    )
    
    # Конвертация в ONNX
    convert_model = KubernetesPodOperator(
        task_id='convert_to_onnx',
        namespace='ml-training',
        image=f"{Variable.get('image_registry')}/model-conversion:latest",
        cmds=['python', '-m', 'src.ml_pipeline.training.onnx_conversion'],
        arguments=[
            '--model-path', '/models/best_model.pth',
            '--output-path', '/models/converted',
            '--input-shape', '1,20'
        ],
        secrets=secrets,
        name='convert-model-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        dag=dag
    )
    
    # Валидация модели
    validate_model = KubernetesPodOperator(
        task_id='validate_model',
        namespace='ml-training',
        image=f"{Variable.get('image_registry')}/model-validation:latest",
        cmds=['python', '-m', 'src.ml_pipeline.validation.validate_model'],
        arguments=[
            '--model-path', '/models/converted/model.onnx',
            '--test-data', '/data/processed/test.csv',
            '--metrics-output', '/reports/validation_metrics.json'
        ],
        secrets=secrets,
        name='validate-model-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        dag=dag
    )
    
    # Сравнение с текущей моделью
    compare_models = KubernetesPodOperator(
        task_id='compare_with_current',
        namespace='ml-training',
        image=f"{Variable.get('image_registry')}/model-comparison:latest",
        cmds=['python', '-m', 'src.ml_pipeline.validation.model_comparison'],
        arguments=[
            '--new-model', '/models/converted/model.onnx',
            '--current-model', '/models/current/model.onnx',
            '--test-data', '/data/processed/test.csv',
            '--output', '/reports/comparison_report.json'
        ],
        secrets=secrets,
        name='compare-models-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        dag=dag
    )
    
    # Регистрация модели в MLflow
    register_model = KubernetesPodOperator(
        task_id='register_model',
        namespace='ml-training',
        image=f"{Variable.get('image_registry')}/model-registration:latest",
        cmds=['python', '-m', 'src.ml_pipeline.registration.register_model'],
        arguments=[
            '--model-path', '/models/converted/model.onnx',
            '--run-id', '{{ task_instance.xcom_pull(task_ids="train_model")["run_id"] }}',
            '--stage', 'Staging',
            '--description', 'Automated retraining {{ ds }}'
        ],
        secrets=secrets,
        name='register-model-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        dag=dag
    )
    
    # A/B тестирование (опционально)
    ab_testing = KubernetesPodOperator(
        task_id='ab_testing',
        namespace='ml-production',
        image=f"{Variable.get('image_registry')}/ab-testing:latest",
        cmds=['python', '-m', 'src.ml_pipeline.testing.ab_test'],
        arguments=[
            '--model-a', 'current',
            '--model-b', 'staging',
            '--traffic-percent', '10',
            '--duration-hours', '24'
        ],
        secrets=secrets,
        name='ab-testing-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        dag=dag
    )
    
    # Продвижение модели в Production
    promote_model = KubernetesPodOperator(
        task_id='promote_to_production',
        namespace='ml-production',
        image=f"{Variable.get('image_registry')}/model-promotion:latest",
        cmds=['python', '-m', 'src.ml_pipeline.deployment.promote_model'],
        arguments=[
            '--model-version', '{{ task_instance.xcom_pull(task_ids="register_model")["model_version"] }}',
            '--validation-report', '/reports/comparison_report.json',
            '--strategy', 'canary',
            '--traffic-percent', '50'
        ],
        secrets=secrets,
        name='promote-model-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        dag=dag
    )
    
    # Обновление мониторинга
    update_monitoring = KubernetesPodOperator(
        task_id='update_monitoring',
        namespace='ml-production',
        image=f"{Variable.get('image_registry')}/monitoring-update:latest",
        cmds=['python', '-m', 'src.ml_pipeline.monitoring.update_reference'],
        arguments=[
            '--new-reference', '/data/processed/train.csv',
            '--model-version', '{{ task_instance.xcom_pull(task_ids="register_model")["model_version"] }}'
        ],
        secrets=secrets,
        name='update-monitoring-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        dag=dag
    )
    
    # Уведомление об успешном завершении
    notify_success = SlackWebhookOperator(
        task_id='notify_success',
        http_conn_id='slack_webhook',
        message="""
✅ *Model Retraining Completed Successfully*
• New model version: {{ task_instance.xcom_pull(task_ids="register_model")["model_version"] }}
• Performance improvement: {{ task_instance.xcom_pull(task_ids="compare_with_current")["improvement"] }}%
• Deployed with canary strategy (50% traffic)
• Monitoring reference data updated
        """,
        username='Airflow ML Pipeline',
        icon_emoji=':rocket:',
        trigger_rule=TriggerRule.ALL_SUCCESS,
        dag=dag
    )
    
    # Уведомление о неудаче
    notify_failure = SlackWebhookOperator(
        task_id='notify_failure',
        http_conn_id='slack_webhook',
        message="""
❌ *Model Retraining Failed*
• DAG Run: {{ run_id }}
• Failed Task: {{ task_instance.task_id }}
• Error: {{ task_instance.state }}
• Check Airflow logs for details
        """,
        username='Airflow ML Pipeline',
        icon_emoji=':x:',
        trigger_rule=TriggerRule.ONE_FAILED,
        dag=dag
    )
    
    # Задача успешного завершения
    end_success = DummyOperator(
        task_id='end_success',
        trigger_rule=TriggerRule.ALL_SUCCESS,
        dag=dag
    )
    
    # Задача неудачного завершения
    end_failure = DummyOperator(
        task_id='end_failure',
        trigger_rule=TriggerRule.ONE_FAILED,
        dag=dag
    )
    
    # Определение зависимостей задач
    start >> [check_drift_task, check_data_task] >> decide_task >> notify_start
    
    # Если принято решение о переобучении
    notify_start >> prepare_data >> train_model >> convert_model >> validate_model
    
    # Параллельные задачи после валидации
    validate_model >> [compare_models, register_model]
    
    # После регистрации и сравнения
    [compare_models, register_model] >> ab_testing >> promote_model >> update_monitoring
    
    # Финальные уведомления
    update_monitoring >> [notify_success, end_success]
    
    # Обработка ошибок
    [train_model, convert_model, validate_model, compare_models, 
     register_model, ab_testing, promote_model, update_monitoring] >> notify_failure >> end_failure

# Вспомогательные функции для работы с XCom
def get_model_version(**context):
    """Получение версии модели из XCom"""
    model_version = context['ti'].xcom_pull(task_ids='register_model', key='model_version')
    return model_version or 'unknown'

def get_validation_results(**context):
    """Получение результатов валидации"""
    validation_results = context['ti'].xcom_pull(task_ids='validate_model')
    return validation_results or {}

# Дополнительные настройки для DAG
dag.doc_md = __doc__
dag.owner_links = {"ml-engineering": "mailto:ml-team@your-bank.com"}

# Добавление тегов для фильтрации
dag.tags = ['mlops', 'retraining', 'credit-scoring', 'production']

# Настройка SLA
dag.sla = timedelta(hours=8)

# Настройка параметров
dag.params = {
    'enable_ab_testing': True,
    'canary_traffic_percent': 50,
    'validation_threshold': 0.02,
    'max_training_time_hours': 4
}