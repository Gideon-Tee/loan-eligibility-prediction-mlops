import os
import sys
import argparse
import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv
load_dotenv()
import mlflow
import mlflow.sklearn

CLEANED_PREFIX = 'cleaned/'
MODELS_PREFIX = 'models/'
S3_BUCKET = 'loan-eligibility-mlops'
TRAIN_FILE = 'loan-train.csv'
TEST_FILE = 'loan-test.csv'

# Get latest timestamp from cleaned/
def get_latest_timestamp(s3, prefix):
    result = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter='/')
    folders = [c['Prefix'].split('/')[-2] for c in result.get('CommonPrefixes', [])]
    if not folders:
        raise Exception('No cleaned dataset folders found in S3!')
    folders = sorted(folders, reverse=True)
    return folders[0]

def download_csv_from_s3(s3, s3_key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

def upload_file_to_s3(s3, local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"Uploaded model to s3://{S3_BUCKET}/{s3_key}")

def main():
    parser = argparse.ArgumentParser(description='Train ML model on cleaned loan eligibility data from S3.')
    parser.add_argument('--timestamp', type=str, help='Timestamp of the cleaned dataset to use (e.g. 20250709_105303)')
    args = parser.parse_args()

    s3 = boto3.client('s3')

    # Determine which cleaned dataset to use
    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = get_latest_timestamp(s3, CLEANED_PREFIX)
        print(f"No timestamp provided. Using latest: {timestamp}")

    cleaned_prefix = f'{CLEANED_PREFIX}dataset-{timestamp}/'
    models_prefix = f'{MODELS_PREFIX}dataset-{timestamp}/'
    os.makedirs('artifacts', exist_ok=True)

    # Download train and test
    train_df = download_csv_from_s3(s3, cleaned_prefix + TRAIN_FILE)
    test_df = download_csv_from_s3(s3, cleaned_prefix + TEST_FILE)

    # Split features/target
    X_train = train_df.drop('Loan_Status', axis=1)
    y_train = train_df['Loan_Status']
    X_test = test_df.drop('Loan_Status', axis=1) if 'Loan_Status' in test_df.columns else test_df
    y_test = test_df['Loan_Status'] if 'Loan_Status' in test_df.columns else None

    # Train models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    best_model = None
    best_acc = 0
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            acc = accuracy_score(y_train, preds)
            print(f"\n{name} - Train Accuracy: {acc:.4f}")
            print(classification_report(y_train, preds))
            mlflow.log_param('model_type', name)
            if name == 'RandomForest':
                mlflow.log_param('n_estimators', model.n_estimators)
            if name == 'LogisticRegression':
                mlflow.log_param('max_iter', model.max_iter)
            mlflow.log_metric('train_accuracy', acc)
            # Evaluate on test if available
            if y_test is not None:
                test_preds = model.predict(X_test)
                test_acc = accuracy_score(y_test, test_preds)
                print(f"{name} - Test Accuracy: {test_acc:.4f}")
                print(classification_report(y_test, test_preds))
                mlflow.log_metric('test_accuracy', test_acc)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = (name, model)
            else:
                if acc > best_acc:
                    best_acc = acc
                    best_model = (name, model)
            # Log model artifact
            model_path = f'artifacts/{name}_model.pkl'
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"MLflow run for {name}: {mlflow.get_artifact_uri('model')}")

    # Save and upload best model
    if best_model:
        model_name, model_obj = best_model
        model_path = f'artifacts/{model_name}_model.pkl'
        joblib.dump(model_obj, model_path)
        s3_key = models_prefix + 'model.pkl'
        upload_file_to_s3(s3, model_path, s3_key)
        print(f"Best model: {model_name} (accuracy: {best_acc:.4f})")
    else:
        print("No model was trained.")

if __name__ == '__main__':
    main() 