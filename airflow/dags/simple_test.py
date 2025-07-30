from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG(
    'simple_test',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
)

task1 = BashOperator(
    task_id='test_task',
    bash_command='echo "Hello World"',
    dag=dag,
)