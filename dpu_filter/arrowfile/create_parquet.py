import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def create_parquet_file(num_rows):
    # Initialize an empty list to store data chunks
    data_chunks = []

    # Set the chunk size to a reasonable value
    chunk_size = 10000

    for start_idx in range(0, num_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, num_rows)

        # Generate a date range chunk
        chunk_data = {
            'timestamp': pd.date_range(start='2023-08-01', periods=end_idx - start_idx, freq='D'),
            'ID': range(101 + start_idx, 101 + end_idx)
        }

        # Append the chunk to the list of data chunks
        data_chunks.append(pd.DataFrame(chunk_data))

    # Concatenate the data chunks into a single DataFrame
    df = pd.concat(data_chunks)

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
num_rows_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
for num_rows in num_rows_list:
    create_parquet_file(num_rows)