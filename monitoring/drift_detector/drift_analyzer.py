import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import *
import boto3
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple
import os

class DriftAnalyzer:
    def __init__(self, s3_bucket: str = 'loan-eligibility-mlops'):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.reports_prefix = 'monitoring/reports/'
        
        # Define column mapping for loan eligibility
        self.column_mapping = ColumnMapping(
            numerical_features=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'],
            categorical_features=['Self_Employed', 'Property_Area'],
            target='Loan_Status'
        )
    
    def load_reference_data(self) -> pd.DataFrame:
        """Load reference dataset (training data)"""
        try:
            # Get latest cleaned training data
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='cleaned/',
                Delimiter='/'
            )
            
            folders = [cp['Prefix'].split('/')[-2] for cp in response.get('CommonPrefixes', [])]
            latest_folder = sorted(folders, reverse=True)[0]
            
            obj = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=f'cleaned/{latest_folder}/loan-train.csv'
            )
            
            reference_data = pd.read_csv(obj['Body'])
            return reference_data
            
        except Exception as e:
            print(f"Error loading reference data: {e}")
            return pd.DataFrame()
    
    def load_current_data(self, days_back: int = 7) -> pd.DataFrame:
        """Load recent prediction data"""
        try:
            from monitoring.data_collector.prediction_logger import PredictionLogger
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            logger = PredictionLogger()
            predictions_df = logger.get_predictions_for_date_range(start_date, end_date)
            
            if predictions_df.empty:
                return pd.DataFrame()
            
            # Extract features from predictions
            features_df = pd.json_normalize(predictions_df['features'])
            features_df['Loan_Status'] = predictions_df['prediction']
            features_df['timestamp'] = predictions_df['timestamp']
            
            return features_df
            
        except Exception as e:
            print(f"Error loading current data: {e}")
            return pd.DataFrame()
    
    def generate_drift_report(self, reference_data: pd.DataFrame, 
                            current_data: pd.DataFrame) -> Optional[Report]:
        """Generate comprehensive drift report"""
        try:
            # Create Evidently report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                ColumnDriftMetric(column_name='ApplicantIncome'),
                ColumnDriftMetric(column_name='CoapplicantIncome'),
                ColumnDriftMetric(column_name='LoanAmount'),
                ColumnDriftMetric(column_name='Self_Employed'),
                ColumnDriftMetric(column_name='Property_Area'),
                DatasetDriftMetric(),
                DatasetMissingValuesMetric()
            ])
            
            # Remove timestamp column for drift analysis
            ref_clean = reference_data.drop(['timestamp'], axis=1, errors='ignore')
            curr_clean = current_data.drop(['timestamp'], axis=1, errors='ignore')
            
            # Run the report
            report.run(
                reference_data=ref_clean,
                current_data=curr_clean,
                column_mapping=self.column_mapping
            )
            
            return report
            
        except Exception as e:
            print(f"Error generating drift report: {e}")
            return None
    
    def save_report(self, report: Report, report_type: str = "drift") -> str:
        """Save report to S3"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save as HTML
            html_key = f"{self.reports_prefix}{report_type}/{timestamp}.html"
            html_content = report.get_html()
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=html_key,
                Body=html_content.encode('utf-8'),
                ContentType='text/html'
            )
            
            # Save as JSON for programmatic access
            json_key = f"{self.reports_prefix}{report_type}/{timestamp}.json"
            json_content = report.json()
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=json_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            
            return f"s3://{self.s3_bucket}/{html_key}"
            
        except Exception as e:
            print(f"Error saving report: {e}")
            return ""
    
    def analyze_drift(self, days_back: int = 7) -> Tuple[bool, dict]:
        """Main method to analyze drift"""
        print("Loading reference data...")
        reference_data = self.load_reference_data()
        
        print("Loading current data...")
        current_data = self.load_current_data(days_back)
        
        if reference_data.empty or current_data.empty:
            return False, {"error": "Insufficient data for drift analysis"}
        
        print(f"Reference data shape: {reference_data.shape}")
        print(f"Current data shape: {current_data.shape}")
        
        # Generate drift report
        report = self.generate_drift_report(reference_data, current_data)
        
        if report is None:
            return False, {"error": "Failed to generate drift report"}
        
        # Save report
        report_url = self.save_report(report, "drift")
        
        # Extract key metrics
        report_dict = json.loads(report.json())
        
        # Check for significant drift
        drift_detected = False
        drift_summary = {}
        
        try:
            # Extract drift metrics from report
            for metric in report_dict.get('metrics', []):
                if metric.get('metric') == 'DatasetDriftMetric':
                    drift_detected = metric.get('result', {}).get('dataset_drift', False)
                    drift_summary['dataset_drift'] = drift_detected
                    drift_summary['drift_share'] = metric.get('result', {}).get('drift_share', 0)
        except Exception as e:
            print(f"Error extracting drift metrics: {e}")
        
        return True, {
            "drift_detected": drift_detected,
            "drift_summary": drift_summary,
            "report_url": report_url,
            "timestamp": datetime.now().isoformat(),
            "data_points": {
                "reference": len(reference_data),
                "current": len(current_data)
            }
        }