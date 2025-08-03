# Cardiovascular Disease Classification Using PCA and XGBoost on AWS SageMaker

## Project Overview

This project detects the presence or absence of cardiovascular disease using a dataset from Kaggle ([dataset link](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)). The workflow applies PCA for dimensionality reduction and XGBoost for classification. The full pipeline is hosted and productionized on **AWS SageMaker** via trained endpoints.

**Key Features:**
- Age, Height, Gender, Smoking, Alcohol Intake, Physical Activity, Systolic & Diastolic Blood Pressure, Cholesterol, Glucose.

**Goal:**  
Build an end-to-end pipeline from data preprocessing and model training to deploying a real-time inference endpoint on SageMaker.

## Prerequisites

1. **Environment Setup**
   ```sh
   conda create -n cardio_vascular_aws_sm python==3.10
   conda activate cardio_vascular_aws_sm
   ```

2. **AWS Setup**
   - AWS Account with Sagemaker and S3 permissions.
   - IAM role with policies for sufficient access.

## 1. Data Preparation & EDA

- **Load Dataset**: Import CSV from Kaggle.
- **Data Cleaning**:
  - Drop `ID` column.
  - Convert `age` from days to years.
  - Remove outliers.
- **Data Visualization**: Use matplotlib/seaborn for distributions and relationships.

## 2. Train-Test Split

- Split dataset typically 80:20 for training and test sets.

## 3. Local Baseline: XGBoost Classification

Before using SageMaker, fit a baseline locally:

```python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```
- **Accuracy**: Typically ~0.74 depending on split/hyperparameters.

## 4. Metrics

- **Confusion Matrix**: Visualize with seaborn heatmap.
- Key metrics:
  - **Accuracy:** $$(TP+TN)/(TP+TN+FP+FN)$$
  - **Misclassification Rate:** $$(FP+FN)/(TP+TN+FP+FN)$$
  - **Precision/Recall/F1**:
    - Precision: $$TP/(TP+FP)$$
    - Recall: $$TP/(TP+FN)$$
    - F1: Harmonic mean of precision & recall.
  - **AUC-ROC**: For binary classifier diagnostic ability.

## 5. Principal Component Analysis (PCA) with SageMaker

PCA reduces dimensions while preserving info:

- **Modes**:
  - Regular: Small data.
  - Randomized: Large data.

**SageMaker Hyperparameters:**
- `feature_dim`: Number of features (here: 11)
- `num_components`: Number of principal components (typically 6, but tunable)
- `mini_batch_size`: Samples per batch, e.g., 100
- `algorithm_mode`: 'regular' or 'randomized'

**Training:**
```python
import sagemaker

# Upload data to S3
bucket = 'cardio-vascular-classification'
prefix = 'cardio_vasuclar_disease_aws_sm'
train_s3_path = f's3://{bucket}/{prefix}/train/pca'

# SageMaker PCA Estimator
pca = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.c4.xlarge',
    output_path=output_location,
    sagemaker_session=sagemaker_session)
pca.set_hyperparameters(feature_dim=11, num_components=6, mini_batch_size=100)

pca.fit({'train': train_s3_path})
```

- PCA can be done in File or Pipe mode in SageMaker.

## 6. Deploy PCA Model

```python
pca_endpoint = pca.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
# Use this endpoint to perform transformation (fit_transform/test_transform) on unseen/test data.
```
- Don’t forget to delete endpoint after inference.

## 7. Training XGBoost with PCA Features (on SageMaker)

- Use transformed dataset from the PCA endpoint.
- Upload train/validation/test splits to S3.

```python
# Prepare SageMaker XGBoost Estimator
xgb = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=output_location,
    sagemaker_session=sagemaker_session)

xgb.set_hyperparameters(
    max_depth=3,
    objective='multi:softmax',
    num_class=2,
    eta=0.1,
    num_round=100,
)

train_input = sagemaker.session.s3_input(train_s3_path, content_type='text/csv')
validation_input = sagemaker.session.s3_input(valid_s3_path, content_type='text/csv')

xgb.fit({'train': train_input, 'validation': validation_input})
```

## 8. Model Evaluation

- Deploy XGBoost to endpoint:
  ```python
  xgb_endpoint = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
  ```
- **Accuracy**: Typically ~0.74 depending on split/hyperparameters.
- Predict on test set and calculate:
  - Confusion Matrix
  - Precision, Recall, F1 Score, ROC/AUC

## 9. Monitoring for Overfitting/Underfitting

- **Overfitting:** High train accuracy, low test accuracy.
- **Underfitting:** Poor performance on both train and test.
- Apply regularization, tune hyperparameters, or adjust model complexity as needed.

## 10. Clean up

- Delete SageMaker endpoints after use to avoid charges:
  ```python
  sagemaker.Session().delete_endpoint(pca_endpoint)
  sagemaker.Session().delete_endpoint(xgb_endpoint)
  ```



## Useful References

- [Kaggle Cardiovascular Disease Data](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- SageMaker Documentation
- XGBoost Documentation

### Endnote

**AWS SageMaker** empowers production-grade ML:  
- **Scale models easily**
- **Consistent reproducibility**
- **Fast from prototype to endpoint**

Stop endpoints after experiments to minimize AWS costs.
