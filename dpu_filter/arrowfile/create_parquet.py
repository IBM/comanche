import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Generate sample data with 6000 rows
num_rows = 200
data = {
    'timestamp': pd.date_range(start='2023-08-01', periods=num_rows, freq='D'),
    'ID': range(101, 101 + num_rows)
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Convert timestamp to Arrow timestamp data type
arrow_timestamps = pa.array(df['timestamp'], type=pa.timestamp('s'))

# Create an Arrow Table
table = pa.Table.from_arrays([arrow_timestamps, pa.array(df['ID'])], names=['timestamp', 'ID'])

# Write the Table to a Parquet file
parquet_file = 'data.parquet'
pq.write_table(table, parquet_file)

print(f"Parquet file '{parquet_file}' created with {num_rows} rows.")
