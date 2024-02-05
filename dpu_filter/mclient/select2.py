#!/usr/bin/env/env python3
import boto3
import gzip

s3 = boto3.client('s3',
                  endpoint_url='https://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

response = s3.get_object(
    Bucket='mycsvbucket',
    Key='sampledata/TotalPopulation.csv.gz'
)

# Decompress the data
with gzip.open(response['Body'], 'rb') as f:
    data = f.read().decode('utf-8')

print(data)
