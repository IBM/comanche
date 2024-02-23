import boto3
import urllib3
import pyarrow.parquet as pq
import io
import pandas as pd

byte_count = 0

# Disable SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

s3 = boto3.client('s3',
                  endpoint_url='https://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

response = s3.select_object_content(
    Bucket='mycsvbucket',
    Key='sampledata/dataStat_100000.parquet',
    ExpressionType='SQL',
    Expression="select * from s3object  where ID < 120",
    InputSerialization={'Parquet': {}},
    OutputSerialization={'JSON': {}},
)

for event in response['Payload']:
    if 'Records' in event:
        records = event['Records']['Payload'].decode('utf-8')
        #byte_count += len(records)
        #print(records)

# Print the byte count
#print(f"Total bytes transferred: {byte_count} bytes")
