import boto3
import urllib3
import time
import json

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Function to read and parse SQL queries from file
def read_sql_queries(file_path):
    queries = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and remove any trailing semicolons
            line = line.strip().strip(';')
            if line:
                # Remove quotes around the query
                line = line.strip('"')
                queries.append(line)
    return queries

# Path to the queries file
queries_file = 'inventoryTPCDS.txt'

# Read SQL queries from file
sql_queries = read_sql_queries(queries_file)

# Configure the S3 client to connect to MinIO
s3 = boto3.client('s3',
                  endpoint_url='http://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

# Process each query
program_start_time = time.time()

for sql_query in sql_queries:
    print(f"Executing query: {sql_query}")

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

    # Measure end time for S3 Select query
    query_end_time = time.time()

    # Process the response and collect all records
    processing_start_time = time.time()
    results = []
    fragment = ""

    for event in response['Payload']:
        if 'Records' in event:
            records = event['Records']['Payload'].decode('utf-8')
            # Accumulate fragments to form complete JSON objects
            fragment += records
            while fragment:
                try:
                    json_record, index = json.JSONDecoder().raw_decode(fragment)
                    results.append(json_record)
                    fragment = fragment[index:].strip()
                except json.JSONDecodeError:
                    # If we get an error, it means we don't have a complete JSON object yet
                    break
        elif 'Stats' in event:
            stats = event['Stats']['Details']
            print(f"Processed {stats['BytesProcessed']} bytes in {stats['BytesReturned']} bytes returned")
        elif 'End' in event:
            print("End of query")

    processing_end_time = time.time()

    # Print the durations
    print(f"S3 Select query execution time: {query_end_time - query_start_time:.2f} seconds")
    print(f"Response processing time: {processing_end_time - processing_start_time:.2f} seconds")

# Measure end time for the whole program
program_end_time = time.time()
print(f"Total program execution time: {program_end_time - program_start_time:.2f} seconds")
