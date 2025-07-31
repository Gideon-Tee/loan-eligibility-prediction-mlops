#!/bin/bash

# LoanFlow Model Monitoring Dashboard Launcher

echo "🚀 Starting LoanFlow Model Monitoring Dashboard..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/loanflow/loan-eligibility-prediction-mlops/monitoring"

# Navigate to monitoring directory
cd /home/ubuntu/loanflow/loan-eligibility-prediction-mlops/monitoring

# Install requirements if needed
if [ ! -f ".requirements_installed" ]; then
    echo "📦 Installing monitoring requirements..."
    pip install -r requirements.txt
    touch .requirements_installed
fi

# Start Streamlit dashboard
echo "🎯 Launching dashboard on http://localhost:8501"
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0