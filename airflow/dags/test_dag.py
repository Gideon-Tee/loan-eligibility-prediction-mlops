from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

dag = DAG(
    'test_simple_dag',
    default_args=default_args,
    description='Simple test DAG',
    catchup=False,
)

test_task = BashOperator(
    task_id='test_task',
    bash_command='echo "Hello World"',
    dag=dag,
)