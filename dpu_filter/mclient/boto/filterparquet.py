import boto3
import urllib3
import pyarrow.parquet as pq
import io
import pandas as pd

# Disable SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

byte_count = 0

s3 = boto3.client('s3',
                  endpoint_url='https://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

response = s3.get_object(
    Bucket='mycsvbucket',
    Key='sampledata/dataStat_10000.parquet'
)

# Read the Parquet data into memory
parquet_data = io.BytesIO(response['Body'].read())

# Measure the length of the response content (bytes)
byte_count = len(parquet_data.getvalue())

# Read the Parquet data as a Pandas DataFrame
df = pq.read_table(parquet_data).to_pandas()

# Filter the DataFrame based on the 'ID' column
filtered_df = df[df['ID'] < 120]

# Print the filtered DataFrame
print(filtered_df)

# Print the byte count
print(f"Total bytes transferred: {byte_count} bytes")
