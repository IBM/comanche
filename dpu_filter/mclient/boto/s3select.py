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
    SELECT SUM(l_extendedprice * l_discount) AS revenue
    FROM S3Object
    WHERE l_shipdate >= '1994-01-01 00:00:00' AND l_shipdate < '1995-01-01 00:00:00'
    AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24000
"""

    #SELECT SUM(l_extendedprice * l_discount) AS revenue
    #FROM S3Object
    #WHERE l_shipdate >= '1994-01-01 00:00:00' AND l_shipdate < '1995-01-01 00:00:00'
    #AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24000
# Measure start time for S3 Select query
query_start_time = time.time()

# Execute the S3 Select query
response = s3.select_object_content(
    Bucket='mycsvbucket',
    Key='sampledata/lineitem_converted.parquet',
    ExpressionType='SQL',
    Expression=sql_query,
    InputSerialization={'Parquet': {}},
    OutputSerialization={'JSON': {}},
)

# Measure end time for S3 Select query
query_end_time = time.time()

# Process the response and collect all records
processing_start_time = time.time()
total_revenue = 0.0

for event in response['Payload']:
    if 'Records' in event:
        records = event['Records']['Payload'].decode('utf-8')
        json_records = json.loads(records)
        print(json_records)  # Debug: Print the JSON record
        total_revenue = json_records['revenue']
    elif 'Stats' in event:
        stats = event['Stats']['Details']
        print(f"Processed {stats['BytesProcessed']} bytes in {stats['BytesReturned']} bytes returned")

processing_end_time = time.time()

# Print the total revenue
print(f"Total revenue: {total_revenue}")

# Measure end time for the whole program
program_end_time = time.time()

# Print the durations
print(f"S3 client initialization time: {s3_client_end_time - s3_client_start_time:.2f} seconds")
print(f"S3 Select query execution time: {query_end_time - query_start_time:.2f} seconds")
print(f"Response processing time: {processing_end_time - processing_start_time:.2f} seconds")
print(f"Total program execution time: {program_end_time - program_start_time:.2f} seconds")

