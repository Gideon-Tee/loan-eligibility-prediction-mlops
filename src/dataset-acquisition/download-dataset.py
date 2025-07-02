import kaggle
import boto3

# Download dataset from Kaggle
kaggle.api.dataset_download_files('dataset-name', path='.', unzip=True)

# Upload to S3
s3 = boto3.client('s3')
s3.upload_file('dataset.csv', 'your-bucket', 'raw/dataset.csv')
