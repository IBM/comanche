#!/usr/bin/env/python3
import boto3
import pyarrow.parquet as pq
import urllib3
import io

# Disable SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

s3 = boto3.client('s3',
                  endpoint_url='https://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

response = s3.get_object(
    Bucket='mycsvbucket',
    Key='sampledata/dataStat_10000.parquet'
)

# Create a BytesIO object from the response content
parquet_data = io.BytesIO(response['Body'].read())

# Get the Parquet file metadata without reading the entire file
parquet_metadata = pq.read_metadata(parquet_data)

# Print the Parquet file metadata
print(parquet_metadata)

