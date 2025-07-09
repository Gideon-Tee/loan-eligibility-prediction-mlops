# import boto3
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# s3 = boto3.client('s3')

# Download cleaned data
# s3.download_file('your-bucket', 'cleaned/cleaned_dataset.csv', 'cleaned_dataset.csv')
data = pd.read_csv('cleaned_dataset.csv')

# Preprocess
X = data.drop('target', axis=1)  # Adjust 'target' to your column name
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with MLflow
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
