# 📊 Monitoring Setup Instructions

## 🚀 EC2 Setup

### 1. Install Service
```bash
# Copy service file
sudo cp monitoring.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable monitoring
```

### 2. Install Dependencies
```bash
# Install monitoring requirements (already added to requirements.txt)
pip install -r requirements.txt
```

### 3. Start Services
```bash
# Use service management script
./manage_services.sh start

# Or individual service
sudo systemctl start monitoring
```

## 🔗 Access Points

- **Airflow**: http://EC2_IP:8080
- **MLflow**: http://EC2_IP:5000  
- **Monitoring Dashboard**: http://EC2_IP:8501

## 📋 Service Status

```bash
# Check all services
./manage_services.sh status

# Individual service
sudo systemctl status monitoring
```

## 🔧 Troubleshooting

### Service Issues
```bash
# Check logs
sudo journalctl -u monitoring -f

# Restart service
sudo systemctl restart monitoring
```

### Dashboard Issues
```bash
# Check Python path
echo $PYTHONPATH

# Manual start for debugging
cd /home/ubuntu/loanflow/loan-eligibility-prediction-mlops
streamlit run monitoring/dashboard/app.py --server.port 8501
```

The monitoring system is now integrated with your existing MLOps infrastructure!