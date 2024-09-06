import boto3
import urllib3
import time
import json
import os

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

# Function to update the result file with query number, timings, and bytes as columns
def update_results_file(query_index, execution_time, processing_time, bytes_processed, bytes_returned):
    result_file = "timing_results_web.txt"
    
    # Check if file exists
    if not os.path.exists(result_file):
        with open(result_file, 'w') as file:
            file.write("Metric\tQuery 1\n")  # Initial header for the first query
            file.write(f"Execution Time (s)\t{execution_time:.3f}\n")
            file.write(f"Processing Time (s)\t{processing_time:.3f}\n")
            file.write(f"Bytes Processed\t{bytes_processed}\n")
            file.write(f"Bytes Returned\t{bytes_returned}\n")
    else:
        # Read the current content of the file
        with open(result_file, 'r') as file:
            lines = file.readlines()

        # Ensure the headers are aligned correctly for subsequent queries
        if query_index == 1:
            lines[0] = lines[0].strip() + f"\tQuery {query_index}\n"
        else:
            lines[0] = lines[0].strip() + f"\tQuery {query_index}\n"
            lines[1] = lines[1].strip() + f"\t{execution_time:.3f}\n"
            lines[2] = lines[2].strip() + f"\t{processing_time:.3f}\n"
            lines[3] = lines[3].strip() + f"\t{bytes_processed}\n"
            lines[4] = lines[4].strip() + f"\t{bytes_returned}\n"

        # Write the updated content back to the file
        with open(result_file, 'w') as file:
            file.writelines(lines)

# Path to the queries file
queries_file = 'inventory.txt'

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

for query_index, sql_query in enumerate(sql_queries, 1):  # Start from Query 1
    print(f"Executing query #{query_index}: {sql_query}")

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

    bytes_processed = 0
    bytes_returned = 0

    for event in response['Payload']:
        if 'Records' in event:
            records = event['Records']['Payload'].decode('utf-8')
            # Accumulate fragments to form complete JSON objects
            fragment += records

            #if len(fragment) > 1024:  # Arbitrary threshold
              #  try:
              #      while True:
                        #json_record, index = json.JSONDecoder().raw_decode(fragment)
                        ##results.append(json_record)
                        #fragment = fragment[index:].lstrip()
               # except json.JSONDecodeError:
                    # Continue accumulating more data if JSON is incomplete
                #    continue
        elif 'Stats' in event:
            stats = event['Stats']['Details']
            bytes_processed = stats['BytesProcessed']
            bytes_returned = stats['BytesReturned']
            print(f"Processed {bytes_processed} bytes, {bytes_returned} bytes returned")
        elif 'End' in event:
            print("End of query")

    processing_end_time = time.time()

    # Query execution and processing time
    execution_time = query_end_time - query_start_time
    processing_time = processing_end_time - processing_start_time

    # Print the durations
    print(f"S3 Select query execution time: {execution_time:.2f} seconds")
    print(f"Response processing time: {processing_time:.2f} seconds")

    # Update result file with query number, execution, processing time, bytes processed, and bytes returned
    update_results_file(query_index, execution_time, processing_time, bytes_processed, bytes_returned)

# Measure end time for the whole program
program_end_time = time.time()
print(f"Total program execution time: {program_end_time - program_start_time:.2f} seconds")
