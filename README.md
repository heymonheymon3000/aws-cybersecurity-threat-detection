# Cybersecurity Threat Detection 

## 🌟 Overview
This project involves building and deploying a Cybersecurity Threat Detection System using Amazon SageMaker. The system identifies anomalous network activity that may indicate cyberattacks, such as DDoS attacks, unauthorized access, or phishing attempts. The machine learning pipeline automates data ingestion, preprocessing, model training, deployment, and inference.

## 🛠️ Services used
* **Amazon SageMaker**: Trains, deploys, and serves the machine learning model. **[Machine Learning]**
* **Amazon S3**: Stores raw network traffic logs, preprocessed data, and model artifacts. **[Storage]**
* **AWS Lambda**: Automates data preprocessing tasks and feature extraction. **[Compute]**
* **Amazon CloudWatch**: Monitors model performance and logs security threats. **[Monitoring]**
* **AWS IAM**: Manages permissions and security policies for accessing AWS services. **[Security]**

#### Create SageMaker role and add AmazonS3FullAccess policy
![alt text](design/sageMakerCybersecurityTrustRelationship.png)
![alt text](design/sageMakerCybersecurityRole.png)
#### Navigate to the sagemaker-ai
![alt text](design/sagemaker-ai-console.png)
#### Create cybersecurity-notebook, click Open Jupyter after status is InsService
![alt text](design/create-cybersecurity-notebook.png)

## ☁️ AWS Architecture


## &rarr; Final Result