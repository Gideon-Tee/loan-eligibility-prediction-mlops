import os
import zipfile
import glob
import boto3
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Kaggle credentials (if not using kaggle.json)
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME', '')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY', '')

import kaggle

# Parameters
KAGGLE_DATASET = 'ravirajsinh45/loan-eligibility-dataset'
LOCAL_RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
DATASET_DIR = os.path.join(LOCAL_RAW_DIR, f'dataset-{TIMESTAMP}')
S3_BUCKET = 'loan-eligibility-mlops'
S3_PREFIX = f'raw/dataset-{TIMESTAMP}/'

os.makedirs(DATASET_DIR, exist_ok=True)

# Download and unzip dataset from Kaggle
print(f"Downloading {KAGGLE_DATASET} to {DATASET_DIR} ...")
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATASET_DIR, unzip=True)

# Find all CSV files
csv_files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))

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
