import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Sample data
data = {
    'timestamp': pd.date_range(start='2023-08-01', periods=10, freq='D'),
    'ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Convert to PyArrow Table
table = pa.Table.from_pandas(df)

# Write the Table to a Parquet file
parquet_file = 'data.parquet'
pq.write_table(table, parquet_file)

print(f"Parquet file '{parquet_file}' created.")
