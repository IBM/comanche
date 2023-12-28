import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import random

def calculate_column_statistics(df):
    # Calculate statistics for the columns using pandas describe
    column_stats = df.describe()

    # Convert column statistics to a dictionary
    column_stats_dict = column_stats.to_dict()

    return column_stats_dict

def create_parquet_file(num_rows):
    # Initialize an empty list to store data chunks
    data_chunks = []

    # Set the chunk size to a reasonable value
    chunk_size = 10000

    for start_idx in range(0, num_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, num_rows)

        # Generate data for the columns
        chunk_data = {
            'timestamp': pd.date_range(start='2023-08-01', periods=end_idx - start_idx, freq='D'),
            'ID': list(range(101 + start_idx, 101 + end_idx)),  # Generate 'ID' column as integers
            'integer_col1': list(np.random.randint(1, 100, size=end_idx - start_idx)),  # Convert to list
            'integer_col2': list(np.random.randint(1, 100, size=end_idx - start_idx)),  # Convert to list
            'string_col': [f'String_{i}' for i in range(101 + start_idx, 101 + end_idx)],
            'Age': [random.randint(18, 80) for _ in range(end_idx - start_idx)],  # Add Age column
            'Name': [f'Name_{i}' for i in range(101 + start_idx, 101 + end_idx)],  # Add Name column
            'Column8': [random.uniform(0, 1) for _ in range(end_idx - start_idx)],  # Add Column8
            'Column9': [random.choice([True, False]) for _ in range(end_idx - start_idx)],  # Add Column9
            'Column10': [random.choice(['A', 'B', 'C']) for _ in range(end_idx - start_idx)]  # Add Column10
        }

        # Append the chunk to the list of data chunks
        data_chunks.append(pd.DataFrame(chunk_data))

    # Concatenate the data chunks into a single DataFrame
    df = pd.concat(data_chunks)

    # Convert timestamp to Arrow timestamp data type
    arrow_timestamps = pa.array(df['timestamp'], type=pa.timestamp('s'))

    # Create Arrow arrays for integer columns
    arrow_int_col1 = pa.array(df['integer_col1'], type=pa.int64())
    arrow_int_col2 = pa.array(df['integer_col2'], type=pa.int64())

    # Create Arrow array for the string column
    arrow_string_col = pa.array(df['string_col'], type=pa.string())

    # Create Arrow arrays for Age and Name columns
    arrow_age_col = pa.array(df['Age'], type=pa.int32())
    arrow_name_col = pa.array(df['Name'], type=pa.string())

    # Create Arrow arrays for the new columns
    arrow_col8 = pa.array(df['Column8'], type=pa.float64())
    arrow_col9 = pa.array(df['Column9'], type=pa.bool_())
    arrow_col10 = pa.array(df['Column10'], type=pa.string())

    # Create Arrow array for the 'ID' column
    arrow_id_col = pa.array(df['ID'], type=pa.int64())

    # Create an Arrow Table with all columns
    table = pa.Table.from_arrays([arrow_timestamps, arrow_id_col, arrow_int_col1, arrow_int_col2, arrow_string_col,
                                  arrow_age_col, arrow_name_col, arrow_col8, arrow_col9, arrow_col10],
                                 names=['timestamp', 'ID', 'integer_col1', 'integer_col2', 'string_col', 'Age', 'Name',
                                        'Column8', 'Column9', 'Column10'])

    # Build the Parquet filename based on the number of rows
    parquet_file = f'dataStat_{num_rows}.parquet'

    # Calculate column statistics
    column_stats = calculate_column_statistics(df)

    # Create a Parquet schema that matches the schema of the Arrow Table
    parquet_schema = pa.schema(table.schema)

    # Create a ParquetWriter and set row group statistics in metadata
    with pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy', write_statistics=True) as writer:
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            writer.write_table(table.slice(start, end))

    print(f"Parquet file '{parquet_file}' created with {num_rows} rows, 10 columns, and row group statistics.")

# Specify the number of rows
num_rows_list = [100000, 1000000, 10000000]
#num_rows_list = [2000000]
for num_rows in num_rows_list:
    create_parquet_file(num_rows)
