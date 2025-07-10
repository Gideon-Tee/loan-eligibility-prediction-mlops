import os
import sys
import argparse
import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support
from dotenv import load_dotenv
load_dotenv()
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import json

# Set MLflow tracking to use project directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")

CLEANED_PREFIX = 'cleaned/'
MODELS_PREFIX = 'models/'
S3_BUCKET = 'loan-eligibility-mlops'
TRAIN_FILE = 'loan-train.csv'  # Use train file for evaluation
TEST_FILE = 'loan-test.csv'

def download_csv_from_s3(s3, s3_key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

def load_model_from_mlflow(model_name, version=None):
    """Load a model from MLflow model registry"""
    client = MlflowClient()
    
    if version is None:
        # Get the latest version
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if not model_versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        version = max([mv.version for mv in model_versions])
    
    model_uri = f"models:/{model_name}/{version}"
    print(f"Loading model: {model_uri}")
    return mlflow.sklearn.load_model(model_uri), version

def get_model_versions(model_name):
    """Get all versions of a model"""
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    return sorted([mv.version for mv in model_versions])

def evaluate_model(model, X_test, y_test, model_name, version):
    """Evaluate a model and return comprehensive metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Calculate ROC AUC if we have probability predictions
    roc_auc = None
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            pass
    
    # Detailed classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'model_name': model_name,
        'version': version,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'detailed_report': report
    }
    
    print(f"\n=== Evaluation Results for {model_name} v{version} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")
    
    return metrics

def compare_models(metrics_a, metrics_b):
    """Compare two models and return the winner"""
    comparison = {
        'model_a': metrics_a,
        'model_b': metrics_b,
        'winner': None,
        'improvement': {}
    }
    
    # Compare key metrics
    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    if metrics_a.get('roc_auc') and metrics_b.get('roc_auc'):
        key_metrics.append('roc_auc')
    
    improvements = {}
    for metric in key_metrics:
        if metric in metrics_a and metric in metrics_b:
            a_val = metrics_a[metric]
            b_val = metrics_b[metric]
            improvement = ((b_val - a_val) / a_val) * 100 if a_val != 0 else 0
            improvements[metric] = improvement
    
    comparison['improvement'] = improvements
    
    # Determine winner based on F1-score (most balanced metric)
    if 'f1_score' in improvements:
        if improvements['f1_score'] > 0:
            comparison['winner'] = metrics_b['model_name']
        else:
            comparison['winner'] = metrics_a['model_name']
    
    return comparison

def promote_model(model_name, version, stage="Production", s3=None):
    """Promote a model to a specific stage and sync to S3"""
    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Successfully promoted {model_name} v{version} to {stage}")
        # S3 sync
        if s3:
            sync_model_stage_to_s3(s3, version, stage)
        return True
    except Exception as e:
        print(f"Error promoting model: {e}")
        return False

def sync_model_stage_to_s3(s3, version, stage):
    """Copy model.pkl and metadata.json from models/version<version>/ to models/<stage>/ in S3"""
    src_prefix = f"models/version{version}/"
    dst_prefix = f"models/{stage.lower()}/"
    for fname in ["model.pkl", "metadata.json"]:
        src_key = src_prefix + fname
        dst_key = dst_prefix + fname
        try:
            s3.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': src_key}, Key=dst_key)
            print(f"Copied {src_key} to {dst_key}")
        except Exception as e:
            print(f"Error copying {src_key} to {dst_key}: {e}")

def main():
    parser = argparse.ArgumentParser(description='A/B evaluation of ML models')
    parser.add_argument('--timestamp', type=str, help='Timestamp of the test dataset to use')
    parser.add_argument('--model-a', type=str, default='LoanEligibilityModel', help='Name of model A')
    parser.add_argument('--version-a', type=int, help='Version of model A (default: previous version)')
    parser.add_argument('--model-b', type=str, help='Name of model B (default: latest trained model)')
    parser.add_argument('--version-b', type=int, help='Version of model B (default: latest)')
    parser.add_argument('--auto-promote', action='store_true', help='Automatically promote the winner to Production')
    args = parser.parse_args()

    s3 = boto3.client('s3')
    client = MlflowClient()

    # Determine test dataset timestamp
    if args.timestamp:
        timestamp = args.timestamp
    else:
        # Get latest cleaned dataset
        result = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=CLEANED_PREFIX, Delimiter='/')
        folders = [c['Prefix'].split('/')[-2] for c in result.get('CommonPrefixes', [])]
        if not folders:
            raise Exception('No cleaned dataset folders found in S3!')
        timestamp = sorted(folders, reverse=True)[0]
        print(f"Using latest dataset: {timestamp}")

    # Download test data
    test_key = f'{CLEANED_PREFIX}dataset-{timestamp}/{TRAIN_FILE}'  # Use train file for evaluation
    test_df = download_csv_from_s3(s3, test_key)
    
    # Prepare test data
    if 'Loan_Status' in test_df.columns:
        X_test = test_df.drop('Loan_Status', axis=1)
        y_test = test_df['Loan_Status']
    else:
        raise ValueError("Evaluation dataset must contain 'Loan_Status' column for evaluation")

    # Load Model A (baseline - previous version)
    model_a_name = args.model_a
    if args.version_a:
        version_a = args.version_a
    else:
        # Get the previous version (second to latest)
        versions = get_model_versions(model_a_name)
        if len(versions) >= 2:
            version_a = versions[-2]  # Previous version
        else:
            version_a = versions[-1]  # Latest if only one version
            print(f"Warning: Only one version available, comparing with itself")
    
    model_a, actual_version_a = load_model_from_mlflow(model_a_name, version_a)
    
    # Load Model B (candidate - latest version)
    if args.model_b:
        model_b_name = args.model_b
        version_b = args.version_b
    else:
        # Use the latest version of the same model
        model_b_name = model_a_name
        version_b = None
    
    model_b, actual_version_b = load_model_from_mlflow(model_b_name, version_b)

    # Evaluate both models
    metrics_a = evaluate_model(model_a, X_test, y_test, model_a_name, actual_version_a)
    metrics_b = evaluate_model(model_b, X_test, y_test, model_b_name, actual_version_b)

    # Compare models
    comparison = compare_models(metrics_a, metrics_b)
    
    print(f"\n=== A/B Comparison Results ===")
    print(f"Model A: {model_a_name} v{actual_version_a}")
    print(f"Model B: {model_b_name} v{actual_version_b}")
    print(f"Winner: {comparison['winner']}")
    print(f"Improvements:")
    for metric, improvement in comparison['improvement'].items():
        print(f"  {metric}: {improvement:+.2f}%")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_dataset_timestamp': timestamp,
        'comparison': comparison,
        'metrics_a': metrics_a,
        'metrics_b': metrics_b
    }
    
    # Save to S3
    results_key = f'evaluation/ab_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=results_key,
        Body=json.dumps(results, indent=2)
    )
    print(f"\nResults saved to s3://{S3_BUCKET}/{results_key}")

    # Auto-promote if requested
    if args.auto_promote and comparison['winner']:
        winner_model = comparison['winner']
        winner_version = actual_version_b if winner_model == metrics_b['model_name'] else actual_version_a
        
        success = promote_model(winner_model, winner_version, "Production", s3=s3)
        if not success:
            print(f"⚠️ MLflow promotion failed, but syncing to S3 anyway.")
            sync_model_stage_to_s3(s3, winner_version, "Production")
        else:
            print(f"✅ Winner {winner_model} v{winner_version} promoted to Production and synced to S3")

    return comparison

if __name__ == '__main__':
    main() 