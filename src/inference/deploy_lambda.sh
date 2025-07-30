#!/bin/bash

# Configuration
AWS_REGION="eu-west-1"
AWS_PROFILE="mlops"
ECR_REPO="loanflow-inference"
LAMBDA_FUNCTION="loanflow-inference"
IMAGE_TAG="latest"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "🏗️ Building Docker image..."
docker build -t ${ECR_REPO}:${IMAGE_TAG} .

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

echo "🏷️ Tagging image..."
docker tag ${ECR_REPO}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}

echo "📤 Pushing to ECR..."
docker push ${ECR_URI}:${IMAGE_TAG}

echo "🚀 Updating Lambda function..."
aws lambda update-function-code \
    --function-name ${LAMBDA_FUNCTION} \
    --image-uri ${ECR_URI}:${IMAGE_TAG} \
    --region ${AWS_REGION}

echo "✅ Deployment complete!"