#!/bin/bash
set -e

PROJECT_DIR="/home/ubuntu/loanflow/loan-eligibility-prediction-mlops"

echo "🚀 Starting deployment..."

# Update code
echo "📥 Updating code..."
cd $PROJECT_DIR
git pull origin main


# Stop services
echo "🛑 Stopping services..."
sudo systemctl stop airflow || true
sudo systemctl stop mlflow || true

# Check if requirements changed
if git diff HEAD~1 HEAD --name-only | grep -q "requirements.txt"; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start services
echo "🚀 Starting services..."
sudo systemctl start airflow
sudo systemctl start mlflow

# Health check
echo "🔍 Health check..."
sleep 15
sudo systemctl is-active airflow && echo "✅ Airflow running" || echo "❌ Airflow failed"
sudo systemctl is-active mlflow && echo "✅ MLflow running" || echo "❌ MLflow failed"

echo "✅ Deployment complete!"