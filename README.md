# Cybersecurity Threat Detection 

## 🌟 Overview
This project involves building and deploying a **Cybersecurity Threat Detection System using Amazon SageMaker**. The system identifies anomalous network activity that may indicate cyberattacks, such as DDoS attacks, unauthorized access, or phishing attempts. The machine learning pipeline automates data ingestion, preprocessing, model training, deployment, and inference.

#### **Key components include:**

* **Data Ingestion & Preprocessing**: Raw network traffic logs are collected, transformed, and feature-engineered to create a structured dataset.
* **Model Training & Evaluation**: An XGBoost model is trained to classify network activity as normal or malicious.
* **Deployment & Inference**: The trained model is deployed as an endpoint to detect real-time security threats.
* **Pipeline Automation**: An end-to-end SageMaker Pipeline automates data transformation, model training, and deployment.

## 🛠️ Services used
* **Amazon SageMaker**: Trains, deploys, and serves the machine learning model. **[Machine Learning]**
* **Amazon S3**: Stores raw network traffic logs, preprocessed data, and model artifacts. **[Storage]**
* **AWS Lambda**: Automates data preprocessing tasks and feature extraction. **[Compute]**
* **Amazon CloudWatch**: Monitors model performance and logs security threats. **[Monitoring]**
* **AWS IAM**: Manages permissions and security policies for accessing AWS services. **[Security]**

## Outline of the steps to complete this project
1. [Preprocess Data and Feature Engineering](#1-preprocess-data-and-feature-engineering)
2. [Training and Testing a Model using XGBoost](#2-training-and-testing-a-model-using-xgboost)
3. [Deploy and Serve the Model](#3-deploy-and-serve-the-model)
4. [Automating with SageMaker Pipelines](#4-automating-with-sagemaker-pipelines)

## 1. Preprocess Data and Feature Engineering
Prepare network traffic data to train a machine learning model for cybersecurity threat detection. Setup AWS environment to fetch public dataset, clean the data, engineer meaningful features, and save the processed data for training. This is a critical step, high-quality input leads to a reliable and accurate model.

**Task to be completed in this step** 
* [Create an IAM Role for SageMaker](#1-create-an-iam-role-for-sagemaker)
* [Set Up Amazon SageMaker Notebook Instance](#2-set-up-amazon-sagemaker-notebook-instance)
* [Download & Upload a Public Dataset to S3](#3-download--upload-a-public-dataset-to-s3)
* [Load & Explore the Dataset in SageMaker](#4-load--explore-the-dataset-in-sagemaker)
* [Clean, Feature Engineer, Encode, and Normalize Data](#5-clean-feature-engineer-encode-and-normalize-data)
* [Save the Preprocessed Data to S3](#6-save-the-preprocessed-data-to-s3)

#### 1. Create an IAM Role for SageMaker
* Navigate to **IAM console** and create a role called **SageMakerCybersecurityRole**
![alt text](design/create-sagemaker-role.png)
* Make sure the **SageMakerCybersecurityRole** has **AmazonSageMakerFullAccess** and **AmazonS3FullAccess**
![alt text](design/create-sagemaker-role-with-S3-policy.png)
* This gives **SageMaker** the permissions required to access the **S3 bucket**

#### 2. Set up Amazon SageMaker Notebook Instance
* Navigate to **Amazon SageMaker AI**

    ![alt text](design/amazon-sagemaker-ai-nav.png)
* Create **notebook instance**
![alt text](design/create-notebook-instance.png)
![alt text](design/create-notebook-instance2.png)
* Create notebook instance and make sure the status InService, then click the Open Jupyter button
![alt text](design/create-notebook-instance3.png)

#### 3. Download & Upload a Public Dataset to S3
* **UNSW_NB15_training-set.csv** is a real-world data public dataset from UNSW-NB15 for threat detection, which contains normal and malicious network activities.
* Navigate to the S3 console and create a bucket.  Leave all other settings as default.
![alt text](design/create-bucket.png)
![alt text](design/create-bucket2.png)
* Navigate into the bucket that was just created and create a folder called raw-data.
![alt text](design/create-bucket3.png)
* Upload **UNSW_NB15_training-set.csv** to the raw-data folder.
![alt text](design/create-bucket4.png)
* Before processing the data, we need to understand it. What features exist, how many records there are, and whether labels are balanced.

    ```python
    import boto3
    import pandas as pd
    import io
    ​  
    # Setup S3 client
    s3_client = boto3.client('s3')
    ​
    # Bucket name
    bucket = 'tparrish-cybersecurity-ml-data'

    # Download the file into memory
    response = s3_client.get_object(Bucket=bucket, Key='raw-data/UNSW_NB15_training-set.csv')
    ​
    # Read it into pandas
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    ​
    # Explore
    print(df.shape)
    print(df.columns)
    print(df.head())
    print(df['label'].value_counts())
    ```

#### 4. Load & Explore the Dataset in SageMaker
* Create a new notebook file

    ![alt text](design/notebook.png)
* Select conda_python3 as the kernel

    ![alt text](design/notebook1.png)
* Rename the file to data_preprocessing.ipynb

    ![alt text](design/notebook2.png)
* File name changed to data_preprocessing.ipynb

    ![alt text](design/notebook3.png)
* Paste the code below in data_preprocessing.ipynb:
    ```python
    import boto3
    import pandas as pd
    import io
    ​  
    # Setup S3 client
    s3_client = boto3.client('s3')
    ​
    # Bucket name
    bucket = 'tparrish-cybersecurity-ml-data'

    # Download the file into memory
    response = s3_client.get_object(Bucket=bucket, Key='raw-data/UNSW_NB15_training-set.csv')
    ​
    # Read it into pandas
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    ​
    # Explore
    print(df.shape)
    print(df.columns)
    print(df.head())
    print(df['label'].value_counts())
    ```

* **'label`** column represents whether the traffic is normal (0) or malicious (1).
* Exploring the data helps identify cleaning and preprocessing needs.

    ![alt text](design/explore-results.png)
    
#### 5. Clean, Feature Engineer, Encode, and Normalize Data
* Prepare the dataset so that our ML models can use it effectively. This step combines data cleaning, feature engineering, categorical encoding, and numerical scaling.

    **1. Drop irrelevant columns**

    - **id** is just a row number and has no predictive value.
    -  **attack_cat** is a more detailed attack type label, but for binary classification (label = malicious or normal), it’s not needed.

    **2. Feature engineering (before encoding & scaling)** </br>
    We create new features from existing ones to help the model detect patterns:

    * **byte_ratio** – Source bytes / (Destination bytes + 1)
        * Helps detect traffic that’s heavily one-sided.
        * +1 prevents division by zero.

    * **is_common_port** – 1 if destination port is 80, 443, or 22
        * Flags traffic over common HTTP/HTTPS/SSH ports.

    * **flow_intensity** – (Source packets + Destination packets) / (Duration + 1e-6)
        * Measures packet rate; useful for spotting floods or spikes.
        * Small number (1e-6) prevents division by zero.

    **3. One-hot encode categorical features**
    - Converts text columns like **proto, service,** and **state** into multiple binary columns.
    - Each unique category becomes its own column with **1 (present) or 0 (absent)**.

    **4. Convert booleans to integers**
    - Some one-hot columns might be **True/False**.
    - We explicitly convert them to **1/0** for compatibility with ML models.

    **5. Scale numerical features**
    - Standardize all numeric columns except **label** so they have mean 0 and standard deviation 1.
    - Prevents features with larger ranges (e.g., byte counts) from dominating smaller-range features (e.g., ratios).

    **6. Sanity checks**
    - Print final dataset shape, column names, a preview of the first rows, and class distribution in **label**. 

 Paste the below code in new cell and Click Run
```python
# --- 1. Drop irrelevant columns ---
df = df.drop(columns=['id', 'attack_cat'])
​
# --- 2. Feature engineering BEFORE encoding/scaling ---
df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
df['is_common_port'] = df['ct_dst_sport_ltm'].isin([80, 443, 22]).astype(int)
df['flow_intensity'] = (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-6)
​
# --- 3. One-hot encode categorical columns ---
categorical_cols = ['proto', 'service', 'state']
df = pd.get_dummies(df, columns=categorical_cols)
​
# --- 4. Convert booleans to ints --
df = df.astype({col: 'int' for col in df.columns if df[col].dtype == 'bool'})
​
# --- 5. Scale numerical features (except label) ---
from sklearn.preprocessing import StandardScaler
​
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('label')
​
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
​
# --- 6. Checks ---
print(df.shape)                                  # Final number of rows & columns
print(df.head())                                 # Preview first few rows
print(df.describe().T[['mean', 'std']])          # Confirm scaling stats
print(df[numerical_cols].mean().round(3))        # Should be ~0
print(df[numerical_cols].std().round(3))         # Should be ~1
```
Run code to clean up data

![alt text](design/cleaning-data.png)

Cleaned up data results

![alt text](design/cleaning-data-results.png)

**Conceptual Notes for Beginners**

* **One-hot encoding**: This takes a column like `proto with values TCP, UDP, and ICMP`, and turns it into 3 new columns - each showing 1 or 0.
* **StandardScaler**: It transforms values so they have a mean of 0 and standard deviation of 1 useful when features have wildly different units (like packet count vs. byte size).

**Tips for Feature Engineering**

* Always divide by `+1` or add a tiny number to avoid division by zero.
* Try plotting distributions of these new features — sometimes it helps visualize how well they separate classes.

#### 6. Save the Preprocessed Data to S3
Dataset is ready, now it can be stored in S3 bucket so it can be accessed later by the ML model.
* Save your processed dataset locally as CSV
* Upload it to your S3 bucket using SageMaker’s session utility
* Paste the below code in new cell and Click Run
    ```python
    import sagemaker
    from sagemaker import get_execution_role
    ​
    # Create SageMaker session and define bucket
    session = sagemaker.Session()
    bucket = 'tparrish-cybersecurity-ml-data'  # Replace with your actual S3 bucket name
    processed_prefix = 'processed-data'        # Folder in S3 to store processed files
    ​
    # Save preprocessed data locally
    df.to_csv('preprocessed_data.csv', index=False)
    ​
    # Upload to S3 inside the 'processed-data/' folder
    s3_path = session.upload_data(
        path='preprocessed_data.csv',
        bucket=bucket,
        key_prefix=processed_prefix
    )
    ​
    print(f"Preprocessed data uploaded to: {s3_path}")
    ```
* Click run

    ![alt text](design/confirmation-file.png)
* Confirm file was uploaded to S3 bucket

    ![alt text](design/confirmation-file-upload.png)

* Validate that ‘processed-data.csv' file got uploaded under the ‘processed-data’ folder

    ![alt text](design/validate-s3-bucket-data.png)

## 2. Training and Testing a Model using XGBoost
Train a machine learning model (using XGBoost) to classify whether a given network activity is normal or malicious, based on the features extracted in Step 1. Use Amazon SageMaker’s built-in XGBoost algorithm, which makes training efficient and scalable.

**Task to be completed in this step** 
* [Load Preprocessed Data from S3](#1-load-preprocessed-data-from-s3)
* [Split Data into Train/Test Sets](#2-split-data-into-traintest-sets)
* [Upload Training and Test Data to S3](#3-upload-training-and-test-data-to-s3)
* [Set Up the XGBoost Training Job](#4-set-up-the-xgboost-training-job)
* [Train the Model](#5-train-the-model)
* [Evaluate Model Performance](#6-evaluate-model-performance)

#### 1. Load Preprocessed Data from S3
#### 2. Split Data into Train/Test Sets
#### 3. Upload Training and Test Data to S3
#### 4. Set Up the XGBoost Training Job
#### 5. Train the Model
#### 6. Evaluate Model Performance

## 3. Deploy and Serve the Model
Deploy the trained model on Amazon SageMaker and expose it as an API endpoint for real-time cybersecurity threat detection.

**Task to be completed in this step** 
* [Create a SageMaker Model from the Trained Model Artifact](#1-create-a-sagemaker-model-from-the-trained-model-artifact)
* [Deploy the Model as a SageMaker Endpoint](#2-deploy-the-model-as-a-sagemaker-endpoint)
* [Test the Deployed Endpoint](#3-test-the-deployed-endpoint)

#### 1. Create a SageMaker Model from the Trained Model Artifact
#### 2. Deploy the Model as a SageMaker Endpoint
#### 3. Test the Deployed Endpoint

## 4. Automating with SageMaker Pipelines
This step connects all the earlier parts of our project into one automated, production-grade ML workflow. It automates the entire machine learning workflow, including data preprocessing, training, evaluation, and deployment using Amazon SageMaker Pipelines.
It transitions our project from: Manual development/testing to Automated pipeline orchestration using Amazon SageMaker Pipelines, Lambda, and EventBridge.


What is Amazon SageMaker Pipelines?

Amazon SageMaker Pipelines is a workflow automation tool that helps:

Streamline data preprocessing, model training, and deployment.
Maintain version control for models.
Automate model retraining when new data arrives.


**Task to be completed in this step** 
* [Define the SageMaker Pipeline Workflow](#1-define-the-sagemaker-pipeline-workflow)
* [Create a SageMaker Pipeline Definition](#2-create-a-sagemaker-pipeline-definition)
* [Trigger the Pipeline Execution](#3-trigger-the-pipeline-execution)
* [Automate Retraining with AWS EventBridge](#4-automate-retraining-with-aws-eventbridge)
* [Test the Automation](#5-test-the-automation)

#### 1. Define the SageMaker Pipeline Workflow
#### 2. Create a SageMaker Pipeline Definition
#### 3. Trigger the Pipeline Execution
#### 4. Automate Retraining with AWS EventBridge
#### 5. Test the Automation






<!-- #### Create SageMaker role and add AmazonS3FullAccess policy
![alt text](design/design/sageMakerCybersecurityTrustRelationship.png)
![alt text](design/design/sageMakerCybersecurityRole.png)
#### Navigate to the sagemaker-ai
![alt text](design/design/sagemaker-ai-console.png)
#### Create cybersecurity-notebook, click Open Jupyter after status is InsService
![alt text](design/design/create-cybersecurity-notebook.png)
#### Create bucket and store UNSW_NB15_training-set.csv in s3://cybersecurity-ml-data-demo/raw-data/
![alt text](design/design/s3-bucket.png)
#### After running
![alt text](design/design/data_preprocessing-output.png)
## ☁️ AWS Architecture


## &rarr; Final Result



pip install --upgrade cmake
!pip install xgboost==3.0.1 -->



