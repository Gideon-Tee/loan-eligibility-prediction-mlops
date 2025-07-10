import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Set MLflow tracking to use project directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")

app = Flask(__name__)

# Global model cache
model_cache = {}

def load_model_from_registry(model_name, version=None, stage=None):
    """Load a model from MLflow model registry"""
    client = MlflowClient()
    
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
    elif stage is not None:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        # Get the latest version
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if not model_versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        latest_version = max([mv.version for mv in model_versions])
        model_uri = f"models:/{model_name}/{latest_version}"
    
    print(f"Loading model: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)

def get_model_info(model_name):
    """Get information about available model versions"""
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    info = {
        'model_name': model_name,
        'versions': []
    }
    
    for mv in model_versions:
        info['versions'].append({
            'version': mv.version,
            'stage': mv.current_stage,
            'status': mv.status,
            'run_id': mv.run_id
        })
    
    return info

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Loan Eligibility Model Server is running'
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        model_name = request.args.get('model_name', 'LoanEligibilityModel')
        info = get_model_info(model_name)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the model"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get model parameters
        model_name = data.get('model_name', 'LoanEligibilityModel')
        version = data.get('version')
        stage = data.get('stage')
        
        # Get features
        features = data.get('features')
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Convert features to DataFrame
        if isinstance(features, list):
            # Single prediction
            df = pd.DataFrame([features])
        elif isinstance(features, dict):
            # Single prediction
            df = pd.DataFrame([features])
        else:
            return jsonify({'error': 'Features must be a list or dict'}), 400
        
        # Load model (with caching)
        cache_key = f"{model_name}_{version}_{stage}"
        if cache_key not in model_cache:
            model_cache[cache_key] = load_model_from_registry(model_name, version, stage)
        
        model = model_cache[cache_key]
        
        # Make prediction
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
        
        # Format response
        result = {
            'prediction': int(prediction[0]),
            'prediction_label': 'Approved' if prediction[0] == 1 else 'Rejected'
        }
        
        if prediction_proba is not None:
            result['confidence'] = float(max(prediction_proba[0]))
            result['probabilities'] = prediction_proba[0].tolist()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Make batch predictions using the model"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get model parameters
        model_name = data.get('model_name', 'LoanEligibilityModel')
        version = data.get('version')
        stage = data.get('stage')
        
        # Get features
        features_list = data.get('features')
        if not features_list or not isinstance(features_list, list):
            return jsonify({'error': 'Features must be a list of records'}), 400
        
        # Convert features to DataFrame
        df = pd.DataFrame(features_list)
        
        # Load model (with caching)
        cache_key = f"{model_name}_{version}_{stage}"
        if cache_key not in model_cache:
            model_cache[cache_key] = load_model_from_registry(model_name, version, stage)
        
        model = model_cache[cache_key]
        
        # Make predictions
        predictions = model.predict(df)
        predictions_proba = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
        
        # Format response
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': int(pred),
                'prediction_label': 'Approved' if pred == 1 else 'Rejected'
            }
            
            if predictions_proba is not None:
                result['confidence'] = float(max(predictions_proba[i]))
                result['probabilities'] = predictions_proba[i].tolist()
            
            results.append(result)
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True) 