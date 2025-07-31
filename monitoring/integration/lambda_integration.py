# Add this to your Lambda handler to enable monitoring

import sys
import os

# Add monitoring path
sys.path.append('/opt/monitoring')

def enhanced_lambda_handler(event, context):
    """Enhanced Lambda handler with monitoring integration"""
    try:
        # Your existing prediction logic
        from lambda_handler import lambda_handler as original_handler
        
        # Get prediction result
        result = original_handler(event, context)
        
        # Log prediction for monitoring (only for successful predictions)
        if result.get('statusCode') == 200:
            try:
                import json
                from monitoring.data_collector.prediction_logger import log_prediction_to_monitoring
                
                # Parse the request and response
                body = json.loads(event.get('body', '{}'))
                response_body = json.loads(result.get('body', '{}'))
                
                if 'features' in body and 'prediction' in response_body:
                    log_prediction_to_monitoring(
                        features=body['features'],
                        prediction=response_body,
                        model_version="v1.0"
                    )
            except Exception as e:
                print(f"Monitoring logging failed: {e}")
                # Don't fail the main prediction
        
        return result
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }