import boto3
from fastparquet.cencoding import from_buffer
import json

# Create an S3 client
s3 = boto3.client('s3',
                  endpoint_url='https://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

# Define the S3 bucket and key for the Parquet file
bucket = 'mycsvbucket'
key = 'sampledata/dataStat_10000.parquet'

# Function to fetch Parquet file metadata
def parquet_metadata_s3(bucket, key, s3):
    metadata = s3.head_object(Bucket=bucket, Key=key)
    content_length = int(metadata['ContentLength'])
    
    end_response = s3.get_object(Bucket=bucket, Key=key, Range=f'bytes={content_length-8}-{content_length}')
    end_content = end_response['Body'].read()
    
    if end_content[-4:] != b'PAR1':
        raise ValueError(f'File at {key} does not look like a Parquet file; magic {end_content[-4:]}')
    
    file_meta_length = int.from_bytes(end_content[:4], byteorder='little')

    file_meta_response = s3.get_object(Bucket=bucket, Key=key,
                                       Range=f'bytes={content_length-8-file_meta_length}-{content_length-8}')
    file_meta_content = file_meta_response['Body'].read()
    
    fmd = from_buffer(file_meta_content, "FileMetaData")
    
    return fmd

# Fetch and parse the Parquet file metadata
parquet_metadata = parquet_metadata_s3(bucket, key, s3)

# Print the raw metadata object
print(parquet_metadata)

