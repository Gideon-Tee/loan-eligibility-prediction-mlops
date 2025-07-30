from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Simple test DAG
dag = DAG(
    'debug_test',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
)

test_task = BashOperator(
    task_id='test_echo',
    bash_command='echo "Test successful"',
    dag=dag,
)