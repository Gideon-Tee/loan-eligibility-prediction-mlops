import json
from lambda_handler import lambda_handler

# Test health check
health_event = {
    'httpMethod': 'GET',
    'path': '/health'
}

print("Testing health check...")
response = lambda_handler(health_event, {})
print(json.dumps(response, indent=2))

# Test prediction
predict_event = {
    'httpMethod': 'POST',
    'path': '/predict',
    'body': json.dumps({
        'features': {
            'Self_Employed': 0,
            'ApplicantIncome': 5720,
            'CoapplicantIncome': 0,
            'LoanAmount': 110,
            'Credit_History': 1.0,
            'Property_Area': 2
        }
    })
}

print("\nTesting prediction...")
response = lambda_handler(predict_event, {})
print(json.dumps(response, indent=2))