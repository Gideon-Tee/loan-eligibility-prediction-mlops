import os
import glob
import boto3
from datetime import datetime
from dotenv import load_dotenv
import kagglehub

# Load environment variables
load_dotenv()

# Parameters
KAGGLE_DATASET = 'vikasukani/loan-eligible-dataset'
S3_BUCKET = 'loan-eligibility-mlops'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
S3_PREFIX = f'raw/dataset-{TIMESTAMP}/'

# Download latest version using kagglehub
print(f"Downloading {KAGGLE_DATASET} using kagglehub ...")
path = kagglehub.dataset_download(KAGGLE_DATASET)
print("Path to dataset files:", path)

# Find all CSV files in the downloaded directory
csv_files = glob.glob(os.path.join(path, '*.csv'))

# Upload CSVs to S3
s3 = boto3.client('s3')
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    s3_key = S3_PREFIX + filename
    print(f"Uploading {csv_file} to s3://{S3_BUCKET}/{s3_key}")
    s3.upload_file(csv_file, S3_BUCKET, s3_key)

print("Upload complete. Files in S3:")
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    s3_key = S3_PREFIX + filename
    print(f"s3://{S3_BUCKET}/{s3_key}")
