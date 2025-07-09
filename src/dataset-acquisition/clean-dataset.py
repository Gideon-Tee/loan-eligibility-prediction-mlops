import os
import sys
import argparse
import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
load_dotenv()


RAW_PREFIX = 'raw/'
CLEANED_PREFIX = 'cleaned/'
S3_BUCKET = 'loan-eligibility-mlops'

# Columns to label encode (skip Loan_Status for test)
CATEGORICAL_COLS = [
    'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'
]
TARGET_COL = 'Loan_Status'

# Columns to convert to int
INT_COLS = ['LoanAmount', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History']

# Columns to drop (if present)
DROP_COLS = ['Dependents', 'Loan_ID', 'Loan_Amount_Term', 'Gender', 'Education', 'Married']

# Files to process
FILES = ['loan-train.csv', 'loan-test.csv']

def get_latest_timestamp(s3, prefix):
    # List all folders under raw/
    result = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter='/')
    folders = [c['Prefix'].split('/')[-2] for c in result.get('CommonPrefixes', [])]
    if not folders:
        raise Exception('No raw dataset folders found in S3!')
    # Sort by timestamp descending
    folders = sorted(folders, reverse=True)
    return folders[0]

def download_csv_from_s3(s3, s3_key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

def upload_csv_to_s3(s3, df, s3_key):
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    s3.upload_fileobj(csv_buffer, S3_BUCKET, s3_key)
    print(f"Uploaded cleaned file to s3://{S3_BUCKET}/{s3_key}")

def clean_dataframe(df, is_train=True):
    # Drop rows with missing values
    df = df.dropna()
    # Label encode categorical columns
    lb = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df.loc[:, col] = lb.fit_transform(df[col])
    if is_train and TARGET_COL in df.columns:
        df.loc[:, TARGET_COL] = lb.fit_transform(df[TARGET_COL])
    # Convert columns to int
    for col in INT_COLS:
        if col in df.columns:
            df.loc[:, col] = df[col].astype(np.int64)
    # Drop unnecessary columns
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop([col], axis=1)
    return df

def main():
    parser = argparse.ArgumentParser(description='Clean raw loan eligibility data from S3 and upload cleaned version.')
    parser.add_argument('--timestamp', type=str, help='Timestamp of the raw dataset to clean (e.g. 20250709_105303)')
    args = parser.parse_args()

    s3 = boto3.client('s3')

    # Determine which raw dataset to use
    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = get_latest_timestamp(s3, RAW_PREFIX)
        print(f"No timestamp provided. Using latest: {timestamp}")

    raw_prefix = f'raw/dataset-{timestamp}/'
    cleaned_prefix = f'cleaned/dataset-{timestamp}/'

    for fname in FILES:
        s3_key = raw_prefix + fname
        try:
            df = download_csv_from_s3(s3, s3_key)
        except Exception as e:
            print(f"Could not download {s3_key}: {e}")
            continue
        is_train = fname == 'loan-train.csv'
        df_clean = clean_dataframe(df, is_train=is_train)
        cleaned_key = cleaned_prefix + fname
        upload_csv_to_s3(s3, df_clean, cleaned_key)

if __name__ == '__main__':
    main()