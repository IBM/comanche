import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def create_parquet_file(num_rows):
    # Generate sample data with the specified number of rows
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

    # Build the Parquet filename based on the number of rows
    parquet_file = f'data_{num_rows}.parquet'

    # Write the Table to a Parquet file
    pq.write_table(table, parquet_file)

    print(f"Parquet file '{parquet_file}' created with {num_rows} rows.")

# Specify the number of rows
num_rows = 100
create_parquet_file(num_rows)
# Specify the number of rows
num_rows = 1000
create_parquet_file(num_rows)
# Specify the number of rows
num_rows = 10000
create_parquet_file(num_rows)
# Specify the number of rows
num_rows = 100000
create_parquet_file(num_rows)
# Specify the number of rows
num_rows = 1000000
create_parquet_file(num_rows)
# Specify the number of rows
num_rows = 10000000
create_parquet_file(num_rows)
# Specify the number of rows
num_rows = 100000000
create_parquet_file(num_rows)
# Specify the number of rows
num_rows = 1000000000
create_parquet_file(num_rows)
