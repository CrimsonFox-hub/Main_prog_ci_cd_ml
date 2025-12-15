# scripts/orchestration/retraining_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPod0perator
from datetime import datetime, timedelta
from airflow.models import Variable

default_args = {
'owner': 'ml-team',
'depends_on_past': False,
'start_date': datetime(2024, 1, 1),
'retries': 2,
'retry_delay': timedelta(minutes-5)
}
with DAG(
'credit_scoring_retraining',
default_args=default_args,
description='Automated retraining pipeline for credit scoring model',
schedule_interval-timedelta(weeks-1),
catchup False,
tags=['mlops', 'retraining']
) as dag:
check_drift = Python0perator(
task_id "check_data_drift",
python_callable-check_drift_task,
provide_context True
)
retrain_model - KubernetesPod0perator(
task_id='retrain_model',
namespace='ml-training',
image-'${IMAGE_REGISTRY}/credit-scoring-trainer: latest',
arguments=['python', 'scripts/model_training/train_models.py'],
name 'retrain-model-pod',
is_delete_operator_pod True,
get_logs=True,
env_vars{
'MLFLOW_TRACKING_URI': Variable.get('mlflow_tracking_uri'),
'DVC_REMOTE_URL': Variable.get('dvc_remote_url')
validate_model = PythonOperator(
task_id="validate_new_model',
python_callable validate_model_task
)
deploy_canary - KubernetesPod0perator(
task_id="deploy_canary_release',
namespace='default',
image='$(IMAGE_REGISTRY}/credit-scoring-deployer: latest',
arguments=["deploy", '--strategy", "canary", '--version', '{{ task_instance.xcom_pull(task_ids="validate_new_model")["model_version"]}}"],
name="deploy-canary-pod"
check_drift >> retrain_model >> validate_model >> deploy_canary