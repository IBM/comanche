import boto3
import urllib3
import pyarrow.parquet as pq
import io
import pandas as pd
import time

byte_count = 0

# Disable SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

s3 = boto3.client('s3',
                  endpoint_url='https://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

start_time = time.time()
download_start_time = time.time()

response = s3.select_object_content(
    Bucket='mycsvbucket',
    Key='sampledata/dataStat_1000000.parquet',
    ExpressionType='SQL',
    Expression="select * from s3object  where ID < 120",
    InputSerialization={'Parquet': {}},
    OutputSerialization={'JSON': {}},
)


download_end_time = time.time()
download_time = download_end_time - download_start_time
print("Download time:", download_time, "seconds")


conversion_start_time = time.time()
for event in response['Payload']:
    if 'Records' in event:
        records = event['Records']['Payload'].decode('utf-8')
        #byte_count += len(records)
        #print(records)
conversion_end_time = time.time()
conversion_time = conversion_end_time - conversion_start_time
print("Conversion time:", conversion_time, "seconds")

end_time = time.time()
full_execution_time = end_time - start_time
print("Full execution time:", full_execution_time, "seconds")


# Print the byte count
#print(f"Total bytes transferred: {byte_count} bytes")
