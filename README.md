# MLOps Mobile Price Classification with AWS SageMaker

A comprehensive end-to-end machine learning operations (MLOps) project demonstrating mobile phone price range classification using AWS SageMaker, Scikit-Learn, and RandomForest classifier.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Details](#model-details)
- [AWS SageMaker Workflow](#aws-sagemaker-workflow)
- [Results](#results)
- [Cost Optimization](#cost-optimization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project demonstrates a complete MLOps workflow for classifying mobile phone price ranges into 4 categories (0-3) based on device specifications. The entire pipeline is built on AWS SageMaker infrastructure, showcasing best practices for:

- Data preparation and exploratory data analysis
- Training machine learning models on cloud infrastructure
- Model deployment and serving
- Endpoint management and cleanup
- Cost optimization using spot instances

## Project Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Local Data     │────▶│   Amazon S3      │────▶│   SageMaker     │
│  Preparation    │     │   (Data Store)   │     │   Training Job  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                            │
                        ┌──────────────────┐              │
                        │   Model          │◀─────────────┘
                        │   Artifacts (S3) │
                        └────────┬─────────┘
                                 │
                        ┌────────▼────────┐
                        │   SageMaker     │
                        │   Endpoint      │
                        └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   Predictions   │
                        └─────────────────┘
```

## Features

- **End-to-End ML Pipeline**: Complete workflow from data preprocessing to model deployment
- **AWS SageMaker Integration**: Leverages SageMaker's managed infrastructure for training and deployment
- **Custom Training Script**: Scikit-Learn based RandomForest implementation with custom training logic
- **S3 Data Management**: Automated data upload and retrieval using Boto3
- **Real-time Inference**: Deployed model endpoint for making predictions
- **Cost Optimization**: Uses spot instances to reduce training costs by up to 90%
- **Comprehensive Logging**: Detailed metrics and classification reports
- **IAM Best Practices**: Proper role-based access control for AWS resources

## Project Structure

```
mlopsmobilepricesagmaker/
│
├── research.ipynb                          # Jupyter notebook with complete workflow
├── script.py                               # SageMaker training script
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── .gitignore                              # Git ignore file
│
├── mob_price_classification_train.csv      # Original training dataset
├── train-V-1.csv                           # Processed training data (85% split)
├── test-V-1.csv                            # Processed testing data (15% split)
└── model.joblib                            # Trained model artifact (local)
```

## Prerequisites

### AWS Requirements

- **AWS Account** with appropriate permissions
- **IAM Role** with the following permissions:
  - AmazonSageMakerFullAccess
  - AmazonS3FullAccess
  - CloudWatchLogsFullAccess
- **AWS CLI** configured with credentials
- **S3 Bucket** for storing training data and model artifacts

### Software Requirements

- Python 3.7 or higher
- Jupyter Notebook/JupyterLab (for running research.ipynb)

### Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `sagemaker` - AWS SageMaker Python SDK
- `scikit-learn` - Machine learning library
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `ipykernel` - Jupyter kernel
- `boto3` - AWS SDK for Python (automatically installed with sagemaker)
- `joblib` - Model serialization (automatically installed with scikit-learn)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mlopsmobilepricesagmaker
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS credentials**:
   ```bash
   aws configure
   ```
   Enter your AWS Access Key ID, Secret Access Key, and default region (e.g., us-east-2)

5. **Update configuration** in [research.ipynb](research.ipynb):
   - Set your S3 bucket name
   - Update IAM role ARN
   - Configure your preferred AWS region

## Dataset

### Source Data
- **File**: `mob_price_classification_train.csv`
- **Target Variable**: `price_range` (4 classes: 0, 1, 2, 3)
- **Features**: Mobile device specifications including:
  - Battery power, RAM, internal memory
  - Camera specifications (front/primary)
  - Display dimensions and resolution
  - Connectivity features (3G, 4G, WiFi, Bluetooth)
  - Physical dimensions (height, width, depth, weight)

### Data Split
- **Training Set**: 85% of data → `train-V-1.csv`
- **Testing Set**: 15% of data → `test-V-1.csv`
- **Random State**: 0 (for reproducibility)

### Data Quality
- No missing values
- Balanced class distribution
- Ready for training without additional preprocessing

## Usage

### 1. Complete Workflow (Recommended)

Open and run [research.ipynb](research.ipynb) which contains the complete end-to-end workflow:

```bash
jupyter notebook research.ipynb
```

The notebook includes:
- Data loading and exploration
- Train-test split
- Data upload to S3
- SageMaker training job creation
- Model deployment
- Inference and predictions
- Endpoint cleanup

### 2. Individual Components

#### Data Preparation
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("mob_price_classification_train.csv")
features = list(df.columns)
label = features.pop(-1)

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[label], test_size=0.15, random_state=0
)
```

#### Training Script
The [script.py](script.py) file contains the SageMaker-compatible training script with:
- Command-line argument parsing for hyperparameters
- Data loading from S3
- RandomForest model training
- Model evaluation and metrics
- Model serialization using joblib

#### SageMaker Training
```python
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role="<YOUR_IAM_ROLE_ARN>",
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    hyperparameters={
        "n_estimators": 100,
        "random_state": 0
    },
    use_spot_instances=True,
    max_run=3600
)

sklearn_estimator.fit({"train": train_s3_path, "test": test_s3_path})
```

#### Model Deployment
```python
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name="Custom-sklearn-model-<timestamp>"
)
```

#### Making Predictions
```python
predictions = predictor.predict(test_data.values.tolist())
print(predictions)
```

#### Cleanup
```python
import boto3
sm_client = boto3.client('sagemaker')
sm_client.delete_endpoint(EndpointName=endpoint_name)
```

## Model Details

### Algorithm
- **Model**: RandomForestClassifier
- **Framework**: Scikit-Learn 0.23-1
- **Type**: Multi-class classification (4 classes)

### Hyperparameters
- `n_estimators`: 100 (number of trees in the forest)
- `random_state`: 0 (for reproducibility)
- `verbose`: 2 (detailed logging)
- `n_jobs`: 1 (single core for consistency)

### Model Serialization
- **Format**: Joblib (.joblib)
- **Function**: `model_fn()` for loading in SageMaker inference
- **Storage**: S3 bucket (model artifacts)

## AWS SageMaker Workflow

### 1. Data Upload to S3
```python
trainpath = sm_session.upload_data(
    path='train-V-1.csv',
    bucket=bucket,
    key_prefix='sagemaker/mobile_price_classification/sklearncontainer'
)
```

### 2. Training Job Configuration
- **Instance Type**: ml.m5.large
- **Spot Instances**: Enabled (up to 90% cost savings)
- **Max Runtime**: 3600 seconds (1 hour)
- **Framework**: Scikit-Learn container
- **Entry Point**: [script.py](script.py)

### 3. Model Training
The training job:
- Reads data from S3
- Trains RandomForest classifier
- Evaluates on test set
- Saves model artifacts to S3
- Logs metrics to CloudWatch

### 4. Model Deployment
- **Instance Type**: ml.m4.xlarge
- **Instance Count**: 1
- **Endpoint Type**: Real-time inference
- **Auto-scaling**: Not configured (single instance)

### 5. Model Artifact Retrieval
```python
artifact = sm_boto3.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]
```

## Results

The model provides:
- **Accuracy Score**: Calculated on test set (15% of data)
- **Classification Report**: Precision, recall, F1-score for each class
- **Confusion Matrix**: Available in training logs
- **CloudWatch Logs**: Detailed training and inference logs

Example output from training:
```
---- METRICS RESULTS FOR TESTING DATA ----
Total Rows are: <test_size>
[TESTING] Model Accuracy is: <accuracy>
[TESTING] Testing Report:
<classification_report>
```

## Cost Optimization

This project implements several cost-saving measures:

1. **Spot Instances**: Training uses spot instances (up to 90% cost reduction)
2. **Instance Selection**: Appropriate instance types for workload
3. **Endpoint Cleanup**: Explicit endpoint deletion to avoid idle charges
4. **S3 Lifecycle Policies**: Can be configured for automatic data cleanup
5. **Training Job Timeout**: Max runtime set to prevent runaway jobs

### Estimated Costs
- **Training**: ~$0.10-0.50 per job (with spot instances)
- **Endpoint**: ~$0.27/hour (ml.m4.xlarge in us-east-2)
- **S3 Storage**: Negligible for this dataset size
- **Data Transfer**: Minimal within same region

**Note**: Always delete endpoints when not in use to avoid ongoing charges.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

---

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)

## Troubleshooting

### Common Issues

1. **IAM Permission Errors**
   - Ensure your IAM role has SageMaker, S3, and CloudWatch permissions
   - Verify the role ARN is correct in the notebook

2. **Region Mismatch**
   - Ensure S3 bucket and SageMaker are in the same region
   - Update region configuration in notebook and boto3 session

3. **Endpoint Costs**
   - Always delete endpoints after use
   - Set up CloudWatch alarms for cost monitoring

4. **Training Job Failures**
   - Check CloudWatch logs for detailed error messages
   - Verify data paths in S3
   - Ensure training script syntax is correct

---

**Project Status**: Active Development