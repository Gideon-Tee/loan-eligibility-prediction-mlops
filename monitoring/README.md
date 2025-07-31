# 📊 LoanFlow Model Monitoring System

A comprehensive model monitoring solution using **Evidently 0.4.1** and **Streamlit** for real-time drift detection and performance tracking.

## 🏗️ Architecture

```
monitoring/
├── data_collector/          # Prediction logging
├── drift_detector/          # Evidently-based drift analysis
├── dashboard/              # Streamlit monitoring UI
├── scheduler/              # Airflow monitoring DAG
├── integration/            # Lambda integration
└── reports/               # Generated monitoring reports
```

## ✨ Features

### 📈 Real-time Monitoring Dashboard
- **Overview Tab**: Key metrics, prediction trends, approval rates
- **Drift Analysis Tab**: Data drift detection with Evidently reports
- **Prediction Analytics Tab**: Feature impact analysis and distributions
- **Alerts Tab**: Automated alerts and model health scoring

### 🔍 Drift Detection
- **Data Drift**: Statistical tests for feature distribution changes
- **Target Drift**: Prediction distribution monitoring
- **Data Quality**: Missing values and data integrity checks
- **Automated Reports**: HTML and JSON reports saved to S3

### ⚠️ Intelligent Alerting
- **Low Prediction Volume**: Usage pattern monitoring
- **Extreme Approval Rates**: Model bias detection
- **Low Confidence Predictions**: Model uncertainty tracking
- **Data Drift Alerts**: Immediate retraining recommendations

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd monitoring
pip install -r requirements.txt
```

### 2. Launch Dashboard
```bash
./run_dashboard.sh
```
Dashboard will be available at: http://localhost:8501

### 3. Deploy Monitoring DAG
```bash
# Copy monitoring DAG to Airflow
cp scheduler/monitoring_dag.py ../airflow/dags/
```

### 4. Update Lambda (Optional)
The Lambda function is already configured to log predictions for monitoring.

## 📊 Dashboard Features

### Overview Tab
- **Total Predictions**: Daily prediction volume
- **Approval Rate**: Loan approval percentage
- **Average Confidence**: Model confidence scores
- **Prediction Trends**: Time series visualizations

### Drift Analysis Tab
- **Drift Status**: Real-time drift detection alerts
- **Feature Drift**: Individual feature drift analysis
- **Distribution Plots**: Current vs reference data comparison
- **Detailed Reports**: Evidently HTML reports

### Prediction Analytics Tab
- **Feature Correlations**: Impact on approval decisions
- **Distribution Analysis**: Feature distributions by outcome
- **Performance Metrics**: Model accuracy tracking

### Alerts Tab
- **Model Health Score**: 0-100 health indicator
- **Alert Categories**: Warning and critical alerts
- **Recommendations**: Actionable insights for model maintenance

## 🔧 Configuration

### Environment Variables
```bash
# AWS Configuration (already set)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=eu-west-1

# S3 Bucket (already configured)
S3_BUCKET=loan-eligibility-mlops
```

### Monitoring Settings
- **Drift Analysis**: Runs daily at 6 AM via Airflow
- **Data Retention**: 30 days of prediction logs
- **Alert Thresholds**: Configurable in dashboard code
- **Report Storage**: S3 bucket under `monitoring/reports/`

## 📈 Monitoring Workflow

1. **Prediction Logging**: Lambda logs all predictions to S3
2. **Daily Analysis**: Airflow runs drift detection daily
3. **Dashboard Updates**: Real-time visualization of metrics
4. **Alert Generation**: Automated alerts for anomalies
5. **Report Storage**: Evidently reports saved to S3

## 🎯 Key Metrics Tracked

### Model Performance
- Prediction volume and trends
- Approval rate variations
- Confidence score distributions
- Response time monitoring

### Data Quality
- Missing value detection
- Feature distribution changes
- Statistical drift measurements
- Data integrity checks

### Business Impact
- User engagement patterns
- Model bias indicators
- Performance degradation alerts
- Retraining recommendations

## 🔍 Troubleshooting

### Dashboard Issues
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/monitoring"

# Restart dashboard
./run_dashboard.sh
```

### Data Loading Issues
- Verify S3 permissions
- Check AWS credentials
- Confirm bucket access

### Drift Analysis Errors
- Ensure sufficient data points (>100 predictions)
- Verify reference data availability
- Check Evidently version compatibility

## 📚 Advanced Usage

### Custom Drift Thresholds
Edit `drift_detector/drift_analyzer.py` to adjust sensitivity:
```python
# Modify drift detection parameters
drift_threshold = 0.1  # Default: 0.05
```

### Additional Metrics
Add custom metrics in `dashboard/app.py`:
```python
# Custom business metrics
custom_metric = calculate_custom_metric(predictions_df)
st.metric("Custom Metric", custom_metric)
```

### Alert Integration
Integrate with external systems in `scheduler/monitoring_dag.py`:
```python
def send_slack_alert(results):
    # Slack integration code
    pass
```

## 🎨 Dashboard Customization

The Streamlit dashboard is fully customizable:
- **Themes**: Modify CSS in `dashboard/app.py`
- **Metrics**: Add custom KPIs and visualizations
- **Layouts**: Adjust tab structure and content
- **Branding**: Update colors, logos, and styling

## 📊 Report Examples

### Drift Report Structure
```json
{
  "drift_detected": true,
  "drift_summary": {
    "dataset_drift": true,
    "drift_share": 0.15
  },
  "report_url": "s3://bucket/monitoring/reports/drift/20250731_120000.html",
  "timestamp": "2025-07-31T12:00:00",
  "data_points": {
    "reference": 1000,
    "current": 150
  }
}
```

## 🚀 Production Deployment

### Scaling Considerations
- **Dashboard**: Deploy on EC2 with load balancer
- **Data Storage**: Use RDS for structured monitoring data
- **Alerting**: Integrate with PagerDuty/Slack
- **Security**: Implement authentication and authorization

### Performance Optimization
- **Caching**: Streamlit caching for expensive operations
- **Data Sampling**: Sample large datasets for analysis
- **Async Processing**: Background drift analysis
- **Resource Limits**: Configure memory and CPU limits

This monitoring system provides comprehensive visibility into your ML model's performance and data quality, enabling proactive maintenance and optimal model performance.