# MLops project End To End Sagemaker Project

## Mobile Price Classification with AWS SageMaker

This project demonstrates a complete machine learning workflow using AWS SageMaker. It covers:

Data preparation and exploration with Pandas and Scikit-Learn
Uploading data to S3 using Boto3 and SageMaker sessions
Training a RandomForest classifier using a custom Scikit-Learn training script on SageMaker
Deploying the trained model as an endpoint
Making predictions and cleaning up the endpoint

## Project Overview

This project aims to classify mobile phone price ranges using a RandomForest classifier. The workflow is built on AWS SageMaker and includes steps for:

- Data ingestion and preprocessing
- Model training on SageMaker using a custom training script
- Model deployment to a SageMaker endpoint
- Inference using the deployed model

## Project Content

The project is organized into several key components:

1. S3 Buckets – Boto3
    Data files (training and testing datasets) are uploaded to Amazon S3 using Boto3. SageMaker retrieves the data from S3 for training.

2. IAM Roles and Users
    Proper AWS IAM roles and permissions are configured to allow SageMaker to create training jobs, access S3, and publish logs to CloudWatch.

3. Complete AWS SageMaker Infrastructure
    The infrastructure covers:

    - Training: A SageMaker training job is launched with a custom Scikit-Learn training script (script.py).
    - Endpoints: The trained model is deployed as an endpoint, and inference requests are sent to this endpoint.
    - Cleanup: The endpoint is deleted after use to avoid unnecessary costs.
    
## Prerequisites

Before running the project, ensure you have the following:

- An AWS account with SageMaker, S3, and CloudWatch permissions.
- AWS CLI configured on your machine.
- Python 3.7+ installed.
- Required Python libraries installed:
    - boto3
    - sagemaker
    - pandas
    - scikit-learn
    - joblib