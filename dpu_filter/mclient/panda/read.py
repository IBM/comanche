import pandas as pd

# Replace these with your MinIO server credentials
access_key = 'minioadmin'
secret_key = 'minioadmin'
endpoint_url = 'https://10.10.10.18:9000'
bucket_name = 'mycsvbucket'

ticker_string = 'dataStat_10000'

# Set the storage options with your credentials
storage_options = {
    'key': access_key,
    'secret': secret_key,
    'endpoint_url': endpoint_url,
    'ca_certs': '/home/nara/public.crt',

}

# Read the Parquet file from the MinIO server
historical_data = pd.read_csv(f's3://{bucket_name}/{ticker_string}.parquet', storage_options=storage_options)

# Display the last 10 rows of the data
historical_data.tail(10)

