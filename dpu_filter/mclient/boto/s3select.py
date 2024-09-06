import boto3
import urllib3
import time
import json

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Measure start time for the whole program
program_start_time = time.time()

# Configure the S3 client to connect to MinIO
s3_client_start_time = time.time()
s3 = boto3.client('s3',
                  endpoint_url='http://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)
s3_client_end_time = time.time()

# Define the SQL query for S3 Select with date and time filtering
sql_query = """
SELECT inv_item_sk, inv_quantity_on_hand FROM S3Object WHERE inv_date_sk <= 2452089
"""

# Measure start time for S3 Select query
query_start_time = time.time()

# Execute the S3 Select query
response = s3.select_object_content(
    Bucket='unencrypted',
    Key='inventory.parquet',
    ExpressionType='SQL',
    Expression=sql_query,
    InputSerialization={'Parquet': {}},
    OutputSerialization={'JSON': {}},
)

# Process the response and collect all records
processing_start_time = time.time()
bytes_returned = 0

records_buffer = ""

for event in response['Payload']:
    if 'Records' in event:
        records = event['Records']['Payload']
        bytes_returned += len(records)
        records_buffer += records.decode('utf-8')
    elif 'Stats' in event:
        stats = event['Stats']['Details']
        print(f"Processed {stats['BytesProcessed']} bytes in {stats['BytesReturned']} bytes returned")

# Split the concatenated JSON records and parse them
#for record in records_buffer.strip().split('\n'):
 #   json_record = json.loads(record)
  #  print(json_record)  # Debug: Print each JSON record

# Measure end time for S3 Select query
query_end_time = time.time()
processing_end_time = time.time()

# Print the total bytes returned
print(f"Total bytes returned: {bytes_returned}")

# Measure end time for the whole program
program_end_time = time.time()

# Print the durations
print(f"S3 client initialization time: {s3_client_end_time - s3_client_start_time:.2f} seconds")
print(f"S3 Select query execution time: {query_end_time - query_start_time:.2f} seconds")
print(f"Response processing time: {processing_end_time - processing_start_time:.2f} seconds")
print(f"Total program execution time: {program_end_time - program_start_time:.2f} seconds")
