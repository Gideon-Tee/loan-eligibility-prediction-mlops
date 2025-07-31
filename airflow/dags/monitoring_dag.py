from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Add monitoring modules to path
sys.path.append('/home/ubuntu/loanflow/loan-eligibility-prediction-mlops/monitoring')

def run_drift_analysis(**context):
    """Run daily drift analysis"""
    from drift_detector.drift_analyzer import DriftAnalyzer
    
    analyzer = DriftAnalyzer()
    success, results = analyzer.analyze_drift(days_back=7)
    
    if success:
        print(f"Drift analysis completed: {results}")
        
        # Send alert if drift detected
        if results.get('drift_detected'):
            print("⚠️ ALERT: Data drift detected!")
            # Here you could integrate with alerting systems
            # send_slack_alert(results)
            # send_email_alert(results)
    else:
        print(f"Drift analysis failed: {results}")
        raise Exception(f"Drift analysis failed: {results}")

def generate_monitoring_report(**context):
    """Generate daily monitoring report"""
    from data_collector.prediction_logger import PredictionLogger
    
    logger = PredictionLogger()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    predictions_df = logger.get_predictions_for_date_range(start_date, end_date)
    
    if not predictions_df.empty:
        total_predictions = len(predictions_df)
        approval_rate = (predictions_df['prediction'] == 1).mean()
        avg_confidence = predictions_df['confidence'].mean() if 'confidence' in predictions_df.columns else 0
        
        print(f"Daily Report - Predictions: {total_predictions}, Approval Rate: {approval_rate:.2%}, Avg Confidence: {avg_confidence:.3f}")
    else:
        print("No predictions found for yesterday")

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_monitoring_pipeline',
    default_args=default_args,
    description='Daily model monitoring and drift detection',
    schedule_interval='0 6 * * *',  # Run daily at 6 AM
    catchup=False,
    tags=['monitoring', 'mlops'],
)

# Task 1: Generate daily monitoring report
daily_report = PythonOperator(
    task_id='generate_daily_report',
    python_callable=generate_monitoring_report,
    dag=dag,
)

# Task 2: Run drift analysis
drift_analysis = PythonOperator(
    task_id='run_drift_analysis',
    python_callable=run_drift_analysis,
    dag=dag,
)

# Task 3: Update dashboard data (optional)
update_dashboard = BashOperator(
    task_id='update_dashboard_cache',
    bash_command='echo "Dashboard cache updated"',
    dag=dag,
)

# Set task dependencies
daily_report >> drift_analysis >> update_dashboard