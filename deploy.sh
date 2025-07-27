#!/bin/bash
set -e

PROJECT_DIR="/home/ubuntu/loan-eligibility"
BACKUP_DIR="/home/ubuntu/backup-$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting deployment..."

# Backup critical data
echo "📦 Creating backup..."
mkdir -p $BACKUP_DIR
cp -r $PROJECT_DIR/mlruns $BACKUP_DIR/ 2>/dev/null || true
cp -r $PROJECT_DIR/airflow/airflow.db $BACKUP_DIR/ 2>/dev/null || true
cp $PROJECT_DIR/.env $BACKUP_DIR/ 2>/dev/null || true

# Stop services
echo "🛑 Stopping services..."
sudo systemctl stop airflow || true
sudo systemctl stop mlflow || true
pkill -f "model_server.py" || true

# Update code
echo "📥 Updating code..."
cd $PROJECT_DIR
git fetch origin
git reset --hard origin/main

# Check if requirements changed
if git diff HEAD~1 HEAD --name-only | grep -q "requirements.txt"; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Restore critical data
echo "🔄 Restoring data..."
cp -r $BACKUP_DIR/mlruns . 2>/dev/null || true
cp -r $BACKUP_DIR/airflow.db airflow/ 2>/dev/null || true
cp $BACKUP_DIR/.env . 2>/dev/null || true

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