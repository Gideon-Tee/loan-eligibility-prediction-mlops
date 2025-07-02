import boto3
import pandas as pd

s3 = boto3.client('s3')

# Download raw data
s3.download_file('your-bucket', 'raw/dataset.csv', 'dataset.csv')
data = pd.read_csv('dataset.csv')

# Clean data (example)
data = data.dropna()

# Upload cleaned data
data.to_csv('cleaned_dataset.csv', index=False)
s3.upload_file('cleaned_dataset.csv', 'your-bucket', 'cleaned/cleaned_dataset.csv')