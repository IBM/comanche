import s3fs
import time
import pyarrow.dataset as ds
import pandas as pd  # Ensure pandas is imported

# Time measurement for full execution
start_time = time.time()

# Configure s3fs to use your MinIO instance
fs = s3fs.S3FileSystem(
    key='minioadmin',  # MinIO access key
    secret='minioadmin',  # MinIO secret key
    client_kwargs={
        'endpoint_url': 'https://10.10.10.18:9000',  # MinIO endpoint URL
        'verify': '/usr/local/share/ca-certificates/public.crt'  # Path to your CA certificate
    },
    config_kwargs={
        's3': {
            'signature_version': 's3v4',  # MinIO recommends using signature version 4
        }
    }
)

# Construct the S3 path to your Parquet file on MinIO
s3_path = 's3://mycsvbucket/sampledata/dataStat_1000000.parquet'

# Time measurement for converting Parquet data to Pandas DataFrame
conversion_start_time = time.time()

# Load the dataset with a filter
dataset = ds.dataset(s3_path, filesystem=fs, format="parquet")
conversion_end_time = time.time()
<<<<<<< HEAD

table = dataset.to_table(filter=ds.field('ID') < 120)


=======
table = dataset.to_table(filter=ds.field('ID') < 120)



>>>>>>> 165cbe02d38de4867a5e604a3af34e0ab07d71c2
# Calculate the conversion time
conversion_time = conversion_end_time - conversion_start_time
print("Conversion time:", conversion_time, "seconds")

# Convert to Pandas DataFrame
df = table.to_pandas()
print(df)

# Time measurement for full execution
end_time = time.time()
full_execution_time = end_time - start_time
print("Full execution time:", full_execution_time, "seconds")
