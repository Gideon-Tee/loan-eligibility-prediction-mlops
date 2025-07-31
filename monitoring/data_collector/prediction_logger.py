import json
import pandas as pd
import boto3
from datetime import datetime
import os
from typing import Dict, Any

class PredictionLogger:
    def __init__(self, s3_bucket: str = 'loan-eligibility-mlops'):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.predictions_prefix = 'monitoring/predictions/'
        
    def log_prediction(self, features: Dict[str, Any], prediction: Dict[str, Any], 
                      model_version: str = "latest", user_id: str = None):
        """Log a single prediction for monitoring"""
        timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'model_version': model_version,
            'user_id': user_id,
            'features': features,
            'prediction': prediction['prediction'],
            'prediction_label': prediction['prediction_label'],
            'confidence': prediction.get('confidence', None),
            'probabilities': prediction.get('probabilities', None)
        }
        
        # Store daily batches
        date_str = timestamp.strftime('%Y-%m-%d')
        s3_key = f"{self.predictions_prefix}{date_str}/predictions.jsonl"
        
        try:
            # Append to existing file or create new
            self._append_to_s3_jsonl(s3_key, log_entry)
            return True
        except Exception as e:
            print(f"Error logging prediction: {e}")
            return False
    
    def _append_to_s3_jsonl(self, s3_key: str, data: Dict):
        """Append data to JSONL file in S3"""
        try:
            # Try to get existing file
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            existing_content = obj['Body'].read().decode('utf-8')
        except:
            existing_content = ""
        
        # Append new line
        new_content = existing_content + json.dumps(data) + '\n'
        
        # Upload back to S3
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=new_content.encode('utf-8'),
            ContentType='application/json'
        )
    
    def get_predictions_for_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve predictions for analysis"""
        predictions = []
        
        # List all prediction files in date range
        response = self.s3_client.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix=self.predictions_prefix
        )
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            if start_date <= key.split('/')[-2] <= end_date:
                try:
                    content = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                    lines = content['Body'].read().decode('utf-8').strip().split('\n')
                    for line in lines:
                        if line:
                            predictions.append(json.loads(line))
                except Exception as e:
                    print(f"Error reading {key}: {e}")
        
        return pd.DataFrame(predictions) if predictions else pd.DataFrame()

# Integration with Lambda function
def log_prediction_to_monitoring(features, prediction, model_version="latest"):
    """Helper function to integrate with Lambda"""
    try:
        logger = PredictionLogger()
        logger.log_prediction(features, prediction, model_version)
    except Exception as e:
        print(f"Monitoring logging failed: {e}")
        # Don't fail the prediction if monitoring fails