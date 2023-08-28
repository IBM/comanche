import pyarrow.parquet as pq

# Path to the Parquet file
parquet_file = 'data.parquet'

# Open the Parquet file
table = pq.read_table(parquet_file)

# Convert the table to a Pandas DataFrame
df = table.to_pandas()

# Print the DataFrame
print(df)
