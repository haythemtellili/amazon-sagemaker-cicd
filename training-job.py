#!/usr/bin/env python
import requests
import os
import pandas as pd

from sagemaker.analytics import TrainingJobAnalytics
import sagemaker
from sagemaker.estimator import Estimator
import boto3
import s3fs

session = sagemaker.Session(boto3.session.Session())

BUCKET_NAME = os.environ['BUCKET_NAME']
PREFIX = os.environ['PREFIX']
REGION = os.environ['AWS_DEFAULT_REGION']
# Replace with your IAM role arn that has enough access (e.g. SageMakerFullAccess)
IAM_ROLE_NAME = os.environ['IAM_ROLE_NAME']
GITHUB_SHA = os.environ['GITHUB_SHA']
ACCOUNT_ID = session.boto_session.client(
    'sts').get_caller_identity()['Account']
# Replace with your desired training instance
training_instance = 'ml.m5.large'

# Replace with your data s3 path
training_data_s3_uri = 's3://{}/{}/boston-housing-training.csv'.format(
    BUCKET_NAME, PREFIX)
validation_data_s3_uri = 's3://{}/{}/boston-housing-validation.csv'.format(
    BUCKET_NAME, PREFIX)


output_folder_s3_uri = 's3://{}/{}/output/'.format(BUCKET_NAME, PREFIX)
source_folder = 's3://{}/{}/source-folders'.format(BUCKET_NAME, PREFIX)
base_job_name = 'boston-housing-model'


# Define estimator object
boston_estimator = Estimator(
    image_uri=f'{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/my-app:latest',
    role=IAM_ROLE_NAME ,
    instance_count=1,
    instance_type=training_instance,
    output_path=output_folder_s3_uri,
    code_location=source_folder,
    base_job_name='boston-housing-model',
    hyperparameters={'nestimators': 70},
    environment={
             "BUCKET_NAME": BUCKET_NAME,
             "PREFIX": PREFIX,
             "GITHUB_SHA": GITHUB_SHA,
             "REGION": REGION,},

    tags=[{"Key": "email",
           "Value": "haythemaws@gmail.com"}])

boston_estimator.fit({'training': training_data_s3_uri,
                      'validation': validation_data_s3_uri}, wait=False)


training_job_name = boston_estimator.latest_training_job.name
hyperparameters_dictionary = boston_estimator.hyperparameters()


report = pd.read_csv(f's3://{BUCKET_NAME}/{PREFIX}/reports.csv')
while(len(report[report['commit_hash']==GITHUB_SHA]) == 0):
    report = pd.read_csv(f's3://{BUCKET_NAME}/{PREFIX}/reports.csv')

res = report[report['commit_hash']==GITHUB_SHA]
metrics_dataframe = res[['Train_MSE', 'Validation_MSE']]

message = (f"## Training Job Submission Report\n\n"
           f"Training Job name: '{training_job_name}'\n\n"
            "Model Artifacts Location:\n\n"
           f"'s3://{BUCKET_NAME}/{PREFIX}/output/{training_job_name}/output/model.tar.gz'\n\n"
           f"Model hyperparameters: {hyperparameters_dictionary}\n\n"
            "See the Logs in a few minute at: "
           f"[CloudWatch](https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={training_job_name})\n\n"
            "If you merge this pull request the resulting endpoint will be avaible this URL:\n\n"
           f"'https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{training_job_name}/invocations'\n\n"
           f"## Training Job Performance Report\n\n"
           f"{metrics_dataframe.to_markdown(index=False)}\n\n"
          )
print(message)

# Write metrics to file
with open('details.txt', 'w') as outfile:
    outfile.write(message)
