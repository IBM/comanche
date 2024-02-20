#!/usr/bin/env/env python3
import boto3
from pyarrow import parquet
import urllib3
import io

# Disable SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


s3 = boto3.client('s3',
                  endpoint_url='http://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

response = s3.get_object(
    Bucket='mycsvbucket',
    Key='sampledata/dataStat_100000.parquet'
)

# Read the Parquet data from the response content
parquet_data = io.BytesIO(response['Body'].read())

parquet_file = parquet.read_table(parquet_data)

print(parquet_file.to_pandas())

