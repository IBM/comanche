import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def read_and_display_parquet(parquet_file):
    # Read the Parquet file
    table = pq.read_table(parquet_file)

    # Convert the Arrow Table to a Pandas DataFrame
    df = table.to_pandas()

    # Display the DataFrame
    print("DataFrame:")
    print(df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py parquet_file")
    else:
        parquet_file = sys.argv[1]
        read_and_display_parquet(parquet_file)