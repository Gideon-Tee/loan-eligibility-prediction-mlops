from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import os
import re
from datetime import datetime, timedelta
import boto3
import mlflow
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def extract_latest_timestamp(**kwargs):
    s3 = boto3.client('s3')
    bucket = 'loan-eligibility-mlops'
    prefix = 'raw/'
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    folders = []
    for cp in result.get('CommonPrefixes', []):
        folder = cp['Prefix'].split('/')[-2]
        if folder.startswith('dataset-'):
            folders.append(folder.replace('dataset-', ''))
    if not folders:
        raise ValueError('No dataset-<timestamp> folders found in S3 raw/')
    latest = sorted(folders, reverse=True)[0]
    kwargs['ti'].xcom_push(key='timestamp', value=latest)
    return latest

def get_timestamp_arg(**kwargs):
    # Used for templating BashOperator command
    return kwargs['ti'].xcom_pull(key='timestamp', task_ids='extract_timestamp')

def get_python():
    # Use the current python executable
    return os.environ.get('PYTHON', 'python3')

def get_script_path(script):
    # Helper to get script path relative to project root
    return os.path.join(PROJECT_ROOT, script)

def_args = {
    'owner': 'airflow',
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 0,
}

default_python = get_python()

with DAG(
    dag_id='loan_eligibility_mlops_pipeline',
    default_args=def_args,
    description='Loan Eligibility ML pipeline orchestrated with Airflow',
    schedule=None,
    catchup=False,
    tags=['mlops', 'loan-eligibility'],
) as dag:
    
    # Task 1: Download dataset
    download_dataset = BashOperator(
        task_id='download_dataset',
        bash_command='{{ params.python }} "{{ params.script }}"',
        params={
            'python': get_python(),
            'script': get_script_path('src/dataset-acquisition/download-dataset.py')
        },
    )

    # Task 2: Extract latest timestamp from data/raw/
    extract_timestamp = PythonOperator(
        task_id='extract_timestamp',
        python_callable=extract_latest_timestamp,
    )

    # Task 3: Clean dataset with timestamp
    clean_dataset = BashOperator(
        task_id='clean_dataset',
        bash_command='{{ params.python }} "{{ params.script }}" --timestamp {{ ti.xcom_pull(task_ids="extract_timestamp", key="timestamp") }}',
        params={
            'python': get_python(),
            'script': get_script_path('src/dataset-acquisition/clean-dataset.py')
        },
    )

    # Task 4: Train model with timestamp
    train_model = BashOperator(
        task_id='train_model',
        bash_command='{{ params.python }} "{{ params.script }}" --timestamp {{ ti.xcom_pull(task_ids="extract_timestamp", key="timestamp") }}',
        params={
            'python': get_python(),
            'script': get_script_path('src/train/train_model.py')
        },
    )

    # Orchestration
    download_dataset >> extract_timestamp >> clean_dataset >> train_model 