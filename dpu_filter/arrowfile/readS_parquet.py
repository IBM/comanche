import pyarrow as pa
import pyarrow.parquet as pq

def read_parquet_file(parquet_file, pa):
    # Read the Parquet file
    table = pq.read_table(parquet_file)

    # Print the schema of the Parquet file
    print(f"Parquet file schema:\n{table.schema}")

    # Access row group statistics (min and max values) for each column
    for column_name in table.column_names:
        column = table[column_name]
        if not pa.types.is_timestamp(column.type):  # Check if it's not a timestamp column
            if isinstance(column, pa.ChunkedArray):
                for chunk in column.iterchunks():
                    if chunk.statistics:  # Access statistics for the chunk
                        print(f"Statistics for column '{column_name}':")
                        print(f"  - Min: {chunk.statistics.min}")
                        print(f"  - Max: {chunk.statistics.max}")
                    else:
                        print(f"No statistics available for column '{column_name}' in chunk.")
            else:
                if column.statistics:  # Access statistics for the column
                    print(f"Statistics for column '{column_name}':")
                    print(f"  - Min: {column.statistics.min}")
                    print(f"  - Max: {column.statistics.max}")
                else:
                    print(f"No statistics available for column '{column_name}'.")
        else:
            print(f"Column '{column_name}' is of timestamp type and does not have statistics.")

# Specify the list of Parquet files to read
parquet_files = ['dataS_10000.parquet']  # Add the filename you want to read here

# Read and process each Parquet file
for parquet_file in parquet_files:
    print(f"Reading Parquet file: {parquet_file}")
    read_parquet_file(parquet_file, pa)
    print("\n")

