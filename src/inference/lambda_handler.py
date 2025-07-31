import json
import pandas as pd
import boto3
import joblib
import numpy
import os

S3_BUCKET = 'loan-eligibility-mlops'
PROD_MODEL_KEY = 'models/production/model.pkl'
LOCAL_MODEL_PATH = '/tmp/model_production.pkl'

# Global model cache
model_cache = {}

def get_cors_headers():
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }

def load_model():
    if 'model' not in model_cache:
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, PROD_MODEL_KEY, LOCAL_MODEL_PATH)
        model_cache['model'] = joblib.load(LOCAL_MODEL_PATH)
    return model_cache['model']

def lambda_handler(event, context):
    try:
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        # Get HTTP method and path
        http_method = event.get('httpMethod', 'POST')
        path = event.get('path', '/predict')
        
        # Health check
        if path == '/health' and http_method == 'GET':
            return {
                'statusCode': 200,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'status': 'healthy',
                    'message': 'LoanFlow Inference Server is running'
                })
            }
        
        # Prediction endpoints
        if path == '/predict' and http_method == 'POST':
            return handle_predict(body)
        elif path == '/predict_batch' and http_method == 'POST':
            return handle_predict_batch(body)
        
        return {
            'statusCode': 404,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': 'Endpoint not found'})
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': str(e)})
        }

def handle_predict(body):
    features = body.get('features')
    if not features:
        return {
            'statusCode': 400,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': 'No features provided'})
        }
    
    # Convert to DataFrame
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        return {
            'statusCode': 400,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': 'Features must be a dict'})
        }
    
    # Load model and predict
    model = load_model()
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
    
    result = {
        'prediction': int(prediction[0]),
        'prediction_label': 'Approved' if prediction[0] == 1 else 'Rejected'
    }
    
    if prediction_proba is not None:
        result['confidence'] = float(max(prediction_proba[0]))
        result['probabilities'] = prediction_proba[0].tolist()
    
    return {
        'statusCode': 200,
        'headers': get_cors_headers(),
        'body': json.dumps(result)
    }

def handle_predict_batch(body):
    features_list = body.get('features')
    if not features_list or not isinstance(features_list, list):
        return {
            'statusCode': 400,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': 'Features must be a list of records'})
        }
    
    df = pd.DataFrame(features_list)
    model = load_model()
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
    
    return {
        'statusCode': 200,
        'headers': get_cors_headers(),
        'body': json.dumps({
            'predictions': results,
            'count': len(results)
        })
    }