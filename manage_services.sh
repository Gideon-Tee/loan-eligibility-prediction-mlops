#!/bin/bash

PROJECT_DIR="/home/ubuntu/loan-eligibility"
export AIRFLOW_HOME=$PROJECT_DIR/airflow

case "$1" in
    start)
        echo "Starting services..."
        sudo systemctl start airflow
        sudo systemctl start mlflow
        cd $PROJECT_DIR/src/inference && nohup python model_server.py > model_server.log 2>&1 &
        echo "Services started"
        ;;
    stop)
        echo "Stopping services..."
        sudo systemctl stop airflow
        sudo systemctl stop mlflow
        pkill -f "model_server.py" || true
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
        pgrep -f "model_server.py" > /dev/null && echo "✅ Model server running" || echo "❌ Model server stopped"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac