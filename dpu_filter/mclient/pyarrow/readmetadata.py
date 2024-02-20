import boto3
import pyarrow.parquet as pq
from pyarrow.fs import S3FileSystem

# Create an S3 client with MinIO configuration
s3 = boto3.client('s3',
                  endpoint_url='http://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)  # Disable SSL certificate verification for MinIO

# S3 bucket name and key
bucket = 'mycsvbucket'
key = 'sampledata/dataStat_100000.parquet'

# Initialize an S3FileSystem object with the S3 client
s3_fs = S3FileSystem()

# S3 path to the Parquet file
# S3 path to the Parquet file
s3_path = "s3://{}/{}".format(bucket, key)


# Open the Parquet file from S3 using the S3FileSystem object
pq_file = pq.ParquetFile(s3_path, filesystem=s3_fs)

# Get metadata information
metadata_info = [
    ["columns:", pq_file.metadata.num_columns],
    ["rows:", pq_file.metadata.num_rows],
    ["row_groups:", pq_file.metadata.num_row_groups]
]

print(metadata_info)

