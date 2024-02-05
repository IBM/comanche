import os
from pyarrow import fs, csv, parquet
import urllib3

# Disable SSL verification for PyArrow S3 connections
os.environ['AWS_CA_BUNDLE'] = ''

# Create an MinIOFileSystem instance pointing to the MinIO server
minio_fs = fs.S3FileSystem(
    endpoint_override='10.10.10.18:9000',  # Address of the MinIO server
    access_key='minioadmin',
    secret_key='minioadmin',
    scheme='https'
    )

# Use minio_fs for reading Parquet files
usernameCSVFile = minio_fs.open_input_file('mycsvbucket/sampledata/TotalPopulation.csv')

# parse CSV data from it
usernameTable = csv.read_csv(usernameCSVFile)
# convert the file to parquet and write it back to MinIO
#parquet.write_to_dataset(
 #     table=usernameTable,
  #    root_path='mycsvbucket/sampledata/username.parquet',
   #   filesystem=minio)
