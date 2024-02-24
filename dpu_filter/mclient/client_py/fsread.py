import pyarrow.parquet as pq
import s3fs

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

# Construct the S3 path to your Parquet file on MinIO
s3_path = 's3://mycsvbucket/sampledata/dataStat_1000000.parquet'

# PyArrow automatically handles efficient data access
dataset = pq.ParquetDataset(s3_path, filesystem=fs)
table = dataset.read()  # Replace 'column1', 'column2' with actual column names

# Now you can work with 'table' as needed, e.g., convert to Pandas DataFrame
df = table.to_pandas()
result = df.query('ID <120')
print(result)