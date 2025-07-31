#!/bin/bash

PROJECT_DIR="/home/ubuntu/loanflow/loan-eligibility-prediction-mlops"
export AIRFLOW_HOME=$PROJECT_DIR/airflow

case "$1" in
    start)
        echo "Starting services..."
        sudo systemctl start airflow
        sudo systemctl start mlflow
        sudo systemctl start monitoring
        echo "Services started"
        ;;
    stop)
        echo "Stopping services..."
        sudo systemctl stop airflow
        sudo systemctl stop mlflow
        sudo systemctl stop monitoring
        echo "Services stopped"
        ;;
    restart)
        $0 stop
        sleep 5
        $0 start
        ;;
    status)
        echo "Service status:"
        sudo systemctl is-active airflow > /dev/null && echo "✅ Airflow running" || echo "❌ Airflow stopped"
        sudo systemctl is-active mlflow > /dev/null && echo "✅ MLflow running" || echo "❌ MLflow stopped"
        sudo systemctl is-active monitoring > /dev/null && echo "✅ Monitoring running" || echo "❌ Monitoring stopped"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac