# Loan Eligibility MLOps Pipeline

A comprehensive MLOps pipeline for loan eligibility prediction using the Kaggle dataset, orchestrated with Apache Airflow and experiment tracking with MLflow.

## 🏗️ Architecture Overview

This project implements a complete MLOps pipeline with the following components:

- **Data Pipeline**: Automated data acquisition, cleaning, and storage in S3
- **Model Training**: ML model training with experiment tracking via MLflow
- **Model Registry**: Centralized model versioning and management
- **A/B Testing**: Automated model comparison and evaluation
- **Model Serving**: REST API for real-time predictions
- **Orchestration**: End-to-end pipeline orchestration with Apache Airflow

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Components](#-pipeline-components)
- [API Documentation](#-api-documentation)
- [Monitoring & Logging](#-monitoring--logging)
- [Deployment](#-deployment)

## ✨ Features

### 🔄 Automated Data Pipeline
- **Data Acquisition**: Automated download of loan eligibility dataset from Kaggle
- **Data Cleaning**: Automated preprocessing including missing value handling, encoding, and feature engineering
- **Data Storage**: Centralized storage in AWS S3 with timestamped versioning
- **Data Validation**: Quality checks and validation before processing

### 🤖 Machine Learning Pipeline
- **Model Training**: Automated training of multiple algorithms (RandomForest, LogisticRegression)
- **Experiment Tracking**: Comprehensive MLflow integration for experiment management
- **Model Selection**: Automatic selection of best performing model based on metrics
- **Model Registration**: Automatic model versioning in MLflow Model Registry

### 🔬 A/B Testing Framework
- **Model Comparison**: Automated comparison between model versions
- **Performance Metrics**: Comprehensive evaluation using accuracy, precision, recall, F1-score, and ROC AUC
- **Automated Promotion**: Automatic promotion of winning models to production
- **Evaluation Reports**: Detailed comparison reports saved to S3

### 🚀 Model Serving
- **REST API**: Flask-based API for real-time predictions
- **Model Caching**: Efficient model loading and caching
- **Batch Predictions**: Support for both single and batch predictions
- **Health Monitoring**: Built-in health checks and monitoring endpoints

### 🎯 Orchestration & Monitoring
- **Airflow DAGs**: End-to-end pipeline orchestration
- **Error Handling**: Robust error handling and retry mechanisms
- **Logging**: Comprehensive logging throughout the pipeline
- **Monitoring**: Real-time monitoring of pipeline execution

## 📁 Project Structure

```
loan-eligibility/
├── airflow/
│   └── dags/
│       └── loan_eligibility_mlops_pipeline.py
├── src/
│   ├── dataset-acquisition/
│   │   ├── download-dataset.py
│   │   └── clean-dataset.py
│   ├── train/
│   │   └── train_model.py
│   ├── evaluation/
│   │   └── ab_evaluation.py
│   └── serving/
│       ├── model_server.py
│       └── requirements.txt
├── mlruns/                          # MLflow tracking data
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Prerequisites

- Python 3.8+
- AWS Account with S3 access
- Kaggle API credentials
- Apache Airflow
- MLflow

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd loan-eligibility
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS and Kaggle credentials
   ```

5. **Initialize Airflow**
   ```bash
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
   ```

## 📖 Usage

### Running the Complete Pipeline

1. **Start Airflow**
   ```bash
   airflow webserver --port 8080 &
   airflow scheduler &
   ```

2. **Trigger the Pipeline**
   - Navigate to Airflow UI: http://localhost:8080
   - Go to DAGs → `loan_eligibility_mlops_pipeline`
   - Click "Trigger DAG" to start the pipeline

### Running Individual Components

#### Data Acquisition
```bash
python src/dataset-acquisition/download-dataset.py
```

#### Data Cleaning
```bash
python src/dataset-acquisition/clean-dataset.py --timestamp 20250709_105303
```

#### Model Training
```bash
python src/train/train_model.py --timestamp 20250709_105303
```

#### A/B Evaluation
```bash
python src/evaluation/ab_evaluation.py --timestamp 20250709_105303 --auto-promote
```

#### Model Serving
```bash
cd src/serving
python model_server.py
```

## 🔄 Pipeline Components

### 1. Data Acquisition (`download-dataset.py`)

**Purpose**: Automatically downloads the loan eligibility dataset from Kaggle and uploads it to S3.

**Features**:
- Uses `kagglehub` for dataset download
- Timestamped folder structure in S3
- Environment variable configuration
- Error handling and logging

**S3 Structure**:
```
s3://loan-eligibility-mlops/raw/dataset-<timestamp>/
├── loan-train.csv
└── loan-test.csv
```

### 2. Data Cleaning (`clean-dataset.py`)

**Purpose**: Preprocesses raw data for machine learning training.

**Processing Steps**:
- Missing value handling
- Categorical encoding (Label Encoding)
- Data type conversions
- Feature engineering
- Quality validation

**S3 Structure**:
```
s3://loan-eligibility-mlops/cleaned/dataset-<timestamp>/
├── loan-train.csv
└── loan-test.csv
```

### 3. Model Training (`train_model.py`)

**Purpose**: Trains multiple ML models and selects the best performer.

**Algorithm Support**:
- RandomForest Classifier
- Logistic Regression

**Features**:
- MLflow experiment tracking
- Comprehensive metrics logging
- Automatic model selection
- Model registration in MLflow Model Registry
- S3 model storage

**MLflow Integration**:
- Experiment tracking with parameters and metrics
- Model artifact logging
- Automatic model registration
- Version management

### 4. A/B Testing (`ab_evaluation.py`)

**Purpose**: Compares model versions and automatically promotes the winner.

**Evaluation Process**:
1. **Model Loading**: Loads two model versions from MLflow registry
2. **Performance Evaluation**: Evaluates both models on test dataset
3. **Metrics Comparison**: Compares accuracy, precision, recall, F1-score, ROC AUC
4. **Winner Selection**: Determines winner based on F1-score improvement
5. **Automated Promotion**: Promotes winning model to Production stage

**Key Features**:
- **Version Comparison**: Automatically compares latest vs previous version
- **Comprehensive Metrics**: Multiple evaluation metrics for thorough comparison
- **Automated Promotion**: Automatic promotion of winning models
- **Results Storage**: Detailed comparison reports saved to S3
- **Flexible Configuration**: Support for custom model versions and evaluation criteria

**Example Usage**:
```bash
# Compare latest vs previous version
python ab_evaluation.py --timestamp 20250709_105303 --auto-promote

# Compare specific versions
python ab_evaluation.py --version-a 1 --version-b 2 --auto-promote
```

### 5. Model Serving (`model_server.py`)

**Purpose**: Provides REST API for real-time model predictions.

**API Endpoints**:
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

**Features**:
- Model caching for performance
- Support for different model versions
- Comprehensive error handling
- JSON-based request/response format

## 🛠️ API Documentation

### Health Check
```bash
curl http://localhost:5050/health
```

### List Models
```bash
curl http://localhost:5050/models?model_name=LoanEligibilityModel
```

### Single Prediction
```bash
curl -X POST http://localhost:5050/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "LoanEligibilityModel",
    "version": 2,
    "features": {
      "Self_Employed": 0,
      "ApplicantIncome": 5720,
      "CoapplicantIncome": 0,
      "LoanAmount": 110,
      "Credit_History": 1.0,
      "Property_Area": 2
    }
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:5050/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "LoanEligibilityModel",
    "features": [
      {
        "Self_Employed": 0,
        "ApplicantIncome": 5720,
        "CoapplicantIncome": 0,
        "LoanAmount": 110,
        "Credit_History": 1.0,
        "Property_Area": 2
      },
      {
        "Self_Employed": 1,
        "ApplicantIncome": 3000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Credit_History": 1.0,
        "Property_Area": 1
      }
    ]
  }'
```

## 📊 Monitoring & Logging

### MLflow Tracking
- **Experiment Tracking**: All training runs logged with parameters and metrics
- **Model Registry**: Centralized model versioning and stage management
- **Artifact Storage**: Model artifacts and evaluation reports stored
- **UI Access**: Access MLflow UI at `http://localhost:5000`

### Airflow Monitoring
- **DAG Execution**: Real-time monitoring of pipeline execution
- **Task Logs**: Detailed logs for each pipeline step
- **Error Handling**: Automatic retry mechanisms and error notifications
- **UI Access**: Access Airflow UI at `http://localhost:8080`

### S3 Storage
- **Data Versioning**: Timestamped data versions for reproducibility
- **Model Storage**: Trained models stored in S3
- **Evaluation Reports**: A/B testing results stored as JSON files

## 🚀 Deployment

### Local Development
```bash
# Start all services
airflow webserver --port 8080 &
airflow scheduler &
mlflow ui --port 5000 &
cd src/serving && python model_server.py &
```

### Production Deployment

#### AWS Deployment
1. **EC2 Instance**: Deploy on AWS EC2 with appropriate security groups
2. **S3 Integration**: Configure S3 bucket for data and model storage
3. **Load Balancer**: Use Application Load Balancer for API endpoints
4. **Auto Scaling**: Implement auto-scaling for high availability

#### Docker Deployment
```dockerfile
# Example Dockerfile for model serving
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/serving/ .
EXPOSE 5050
CMD ["python", "model_server.py"]
```

## 🔧 Configuration

### Environment Variables
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Kaggle Configuration
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# S3 Configuration
S3_BUCKET=loan-eligibility-mlops

# MLflow Configuration
MLFLOW_TRACKING_URI=file:///path/to/mlruns
```

### Airflow Configuration
- **DAG Schedule**: Configure pipeline execution frequency
- **Retry Logic**: Set retry attempts for failed tasks
- **Resource Limits**: Configure CPU/memory limits for tasks
- **Email Notifications**: Set up email alerts for pipeline failures

## 📈 Performance Metrics

### Model Performance
- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision for loan approval predictions
- **Recall**: Recall for loan approval predictions
- **F1-Score**: Balanced metric for model performance
- **ROC AUC**: Area under ROC curve for probability predictions

### Pipeline Performance
- **Execution Time**: Total pipeline execution time
- **Data Processing Time**: Time for data cleaning and preparation
- **Model Training Time**: Time for model training and evaluation
- **API Response Time**: Model serving response times

## 🔍 Troubleshooting

### Common Issues

1. **MLflow Model Not Visible**
   - Check MLflow tracking URI configuration
   - Verify model registration in training script
   - Check MLflow UI for registered models

2. **Airflow Task Failures**
   - Check task logs in Airflow UI
   - Verify environment variables and paths
   - Check S3 permissions and connectivity

3. **A/B Evaluation Issues**
   - Ensure test dataset has target variable
   - Check model version availability
   - Verify evaluation metrics calculation

4. **API Connection Issues**
   - Check model server is running
   - Verify port configuration
   - Check model loading and caching


## 🙏 Acknowledgments

- Kaggle for providing the loan eligibility [dataset](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset/data)
- Apache Airflow for workflow orchestration
- MLflow for experiment tracking and model management
- AWS for cloud infrastructure and S3 storage
