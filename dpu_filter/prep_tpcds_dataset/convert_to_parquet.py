import pandas as pd
import os

# Directory where .dat files are located
input_dir = '/mnt/sda4/tpcds_sf10'
# Directory to save Parquet files
output_dir = '/mnt/sda4/parquet_tpcds_sf10'
os.makedirs(output_dir, exist_ok=True)

# List of specific TPC-DS tables to convert
tables = [
    'catalog_returns', 'catalog_sales', 'customer_demographics', 
    'customer', 'inventory', 'store_returns', 'store_sales'
]

for table in tables:
    file_path = os.path.join(input_dir, f'{table}.dat')
    df = pd.read_csv(file_path, delimiter='|', index_col=False)
    df = df.iloc[:, :-1]  # Remove the last column which is empty due to the delimiter
    table_path = os.path.join(output_dir, f'{table}.parquet')
    df.to_parquet(table_path, engine='pyarrow')

print("Conversion to Parquet completed.")

