import pyarrow.parquet as pq
import s3fs
import time
# Time measurement for full execution
start_time = time.time()

# Time measurement for download
download_start_time = time.time()
# Configure s3fs to use your MinIO instance
fs = s3fs.S3FileSystem(
    key='minioadmin',  # MinIO access key
    secret='minioadmin',  # MinIO secret key
    client_kwargs={
        'endpoint_url': 'https://10.10.10.18:9000',  # MinIO endpoint URL
        'verify': False  # If your MinIO server uses self-signed certificates, you might need this
    },
    config_kwargs={
        's3': {
            'signature_version': 's3v4',  # MinIO recommends using signature version 4
        }
    }
)

# Time measurement for download
download_end_time = time.time()

# Calculate the download time
download_time = download_end_time - download_start_time
print("Download time:", download_time, "seconds")

# Time measurement for converting Parquet data to Pandas DataFrame
conversion_start_time = time.time()
# Construct the S3 path to your Parquet file on MinIO
s3_path = 's3://mycsvbucket/sampledata/dataStat_1000000.parquet'

# PyArrow automatically handles efficient data access
dataset = pq.ParquetDataset(s3_path, filesystem=fs)

table = dataset.read()  # Replace 'column1', 'column2' with actual column names
conversion_end_time = time.time()


# Calculate the conversion time
conversion_time = conversion_end_time - conversion_start_time
print("Conversion time:", conversion_time, "seconds")

# Now you can work with 'table' as needed, e.g., convert to Pandas DataFrame
df = table.to_pandas()
result = df.query('ID <120')
print(result)


# Time measurement for full execution
end_time = time.time()
full_execution_time = end_time - start_time
print("Full execution time:", full_execution_time, "seconds")