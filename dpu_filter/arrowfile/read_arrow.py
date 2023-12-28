import pyarrow.parquet as pq

def read_parquet_metadata(parquet_file):
    # Read the Parquet file metadata
    metadata = pq.read_metadata(parquet_file)

    # Print the metadata, which includes row group information
    print(f"Parquet file metadata:\n{metadata}")

# Specify the list of Parquet files to read
parquet_files = ['dataS_10000.parquet']  # Add the filename you want to read here

# Read and process each Parquet file's metadata
for parquet_file in parquet_files:
    print(f"Reading Parquet file metadata: {parquet_file}")
    read_parquet_metadata(parquet_file)
    print("\n")

