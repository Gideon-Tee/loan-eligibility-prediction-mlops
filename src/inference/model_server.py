import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()
import boto3
import joblib

S3_BUCKET = 'loan-eligibility-mlops'
PROD_MODEL_KEY = 'models/production/model.pkl'
PROD_METADATA_KEY = 'models/production/metadata.json'
LOCAL_MODEL_PATH = '/tmp/model_production.pkl'
LOCAL_METADATA_PATH = '/tmp/metadata_production.json'

app = Flask(__name__)

# Global model cache
model_cache = {}
metadata_cache = {}

def download_from_s3(s3, key, local_path):
    s3.download_file(S3_BUCKET, key, local_path)

def load_production_model():
    if 'model' not in model_cache:
        s3 = boto3.client('s3')
        download_from_s3(s3, PROD_MODEL_KEY, LOCAL_MODEL_PATH)
        model_cache['model'] = joblib.load(LOCAL_MODEL_PATH)
    return model_cache['model']

def load_production_metadata():
    if 'metadata' not in metadata_cache:
        s3 = boto3.client('s3')
        try:
            download_from_s3(s3, PROD_METADATA_KEY, LOCAL_METADATA_PATH)
            with open(LOCAL_METADATA_PATH, 'r') as f:
                metadata_cache['metadata'] = json.load(f)
        except Exception:
            metadata_cache['metadata'] = None
    return metadata_cache['metadata']

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Loan Eligibility Model Server is running'
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    metadata = load_production_metadata()
    if metadata:
        return jsonify(metadata)
    else:
        return jsonify({'error': 'No metadata found for production model.'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        features = data.get('features')
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        # Convert features to DataFrame
        if isinstance(features, list):
            df = pd.DataFrame([features])
        elif isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            return jsonify({'error': 'Features must be a list or dict'}), 400
        # Load model
        model = load_production_model()
        # Make prediction
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
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
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        features_list = data.get('features')
        if not features_list or not isinstance(features_list, list):
            return jsonify({'error': 'Features must be a list of records'}), 400
        df = pd.DataFrame(features_list)
        model = load_production_model()
        predictions = model.predict(df)
        predictions_proba = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
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