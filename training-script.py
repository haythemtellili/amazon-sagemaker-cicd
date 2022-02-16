#!/usr/bin/env python
import os
import joblib
import requests
import json
from datetime import datetime, timezone


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import boto3
import botocore


def update_report_file(metrics_dictionary: dict, hyperparameters: dict,
                       commit_hash: str, training_job_name: str,
                       prefix: str, bucket_name: str,) -> None:
    """This funtion update the report file located in the S3 bucket according to the provided metrics
    if report file doesn't exist, it will create a template based on metrics_dictionary schema and upload it to S3
    Args:
        metrics_dictionary (dict): the training job metrics with this format: {"Metric_1_Name": "Metric_1_Value", ...}
        hyperparameters (dict): the training job hyperparameters with this format: {"Hyperparameter_1_Name": "Hyperparameter_1_Value", ...}
        commit_hash (str): the 7 digit hash of the commit that started this training job
        training_job_name (str): name of the current training job
        prefix (str): name of the folder in the S3 bucket
        bucket_name (str): name of the S3 bucket
    Returns:
        None
    """
    object_key = f'{prefix}/reports.csv'

    s3 = boto3.resource('s3')

    try:
        s3.Bucket(bucket_name).download_file(object_key, 'reports.csv')

        # Load reports df
        reports_df = pd.read_csv('reports.csv')

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            columns = ['date_time', 'hyperparameters', 'commit_hash',
                       'training_job_name'] + list(metrics_dictionary.keys())
            pd.DataFrame(columns=columns).to_csv('./reports.csv', index=False)

            # Upload template reports df
            s3.Bucket(bucket_name).upload_file('./reports.csv', object_key)

            # Load reports df
            reports_df = pd.read_csv('./reports.csv')

        else:
            raise

    # Add new report to reports.csv
    # Use UTC time to avoid timezone heterogeneity
    date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Add new row
    new_row = dict({'date_time': date_time, 'hyperparameters': json.dumps(hyperparameters), 'commit_hash': commit_hash, 'training_job_name': training_job_name},
                   **metrics_dictionary)
    new_report = pd.DataFrame(new_row, index=[0])
    reports_df = reports_df.append(new_report)

    # Upload new reports dataframe
    reports_df.to_csv('./reports.csv', index=False)
    s3.Bucket(bucket_name).upload_file('./reports.csv', object_key)


# Define main training function
def main():
    with open('/opt/ml/input/config/hyperparameters.json', 'r') as json_file:
        hyperparameters = json.load(json_file)
        print(hyperparameters)

    with open('/opt/ml/input/config/inputdataconfig.json', 'r') as json_file:
        inputdataconfig = json.load(json_file)
    print(inputdataconfig)

    with open('/opt/ml/input/config/resourceconfig.json', 'r') as json_file:
        resourceconfig = json.load(json_file)
    print(resourceconfig)

    # Load Data
    training_data_path = '/opt/ml/input/data/training'
    validation_data_path = '/opt/ml/input/data/validation'
    training_data = pd.read_csv(os.path.join(
        training_data_path, 'boston-housing-training.csv'))
    validation_data = pd.read_csv(os.path.join(
        validation_data_path, 'boston-housing-validation.csv'))

    print(training_data)
    print(validation_data)

    X_train, y_train = training_data.iloc[:,
                                          1:].values, training_data.iloc[:, :1].values
    X_val, y_val = validation_data.iloc[:,
                                        1:].values, validation_data.iloc[:, :1].values

    # Fit the model
    n_estimators = int(hyperparameters['nestimators'])
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Evaluate model
    train_mse = mean_squared_error(model.predict(X_train), y_train)
    val_mse = mean_squared_error(model.predict(X_val), y_val)

    metrics_dictionary = {'Train_MSE': train_mse,
                          'Validation_MSE': val_mse,}
    metrics_dataframe = pd.DataFrame(metrics_dictionary, index=[0])

    print(metrics_dictionary)
    
    # Save the model
    model_path = '/opt/ml/model'
    model_path_full = os.path.join(model_path, 'model.joblib')
    joblib.dump(model, model_path_full)

    
    # Update the Report File
    REGION = os.environ['REGION']
    PREFIX = os.environ['PREFIX']
    BUCKET_NAME = os.environ['BUCKET_NAME']
    GITHUB_SHA = os.environ['GITHUB_SHA']
    TRAINING_JOB_NAME = os.environ['TRAINING_JOB_NAME']

    update_report_file(metrics_dictionary=metrics_dictionary, hyperparameters=hyperparameters,
                       commit_hash=GITHUB_SHA, training_job_name=TRAINING_JOB_NAME, prefix=PREFIX, bucket_name=BUCKET_NAME)

if __name__ == '__main__':
    main()