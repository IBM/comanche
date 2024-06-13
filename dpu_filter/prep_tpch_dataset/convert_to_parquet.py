import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Function to load and convert TPC-H .tbl files to Parquet with specified row group size in records
def convert_to_parquet(file_path, columns, output_dir, row_group_size=100000):
    file_name = os.path.basename(file_path).split('.')[0]

    # Read the entire .tbl file into a DataFrame
    df = pd.read_csv(file_path, delimiter='|', header=None, names=columns)

    # Convert specific columns to appropriate types
    if 'l_shipdate' in df.columns:
        df['l_shipdate'] = pd.to_datetime(df['l_shipdate'])

    # Convert DataFrame to Apache Arrow table
    table = pa.Table.from_pandas(df)

    # Write Arrow table to Parquet file with specified row group size
    parquet_file_path = os.path.join(output_dir, f'{file_name}.parquet')
    pq.write_table(table, parquet_file_path, row_group_size=row_group_size)

    # Print number of row groups and the size of each row group in the generated Parquet file
    parquet_file = pq.ParquetFile(parquet_file_path)
    print(f'File: {parquet_file_path}, Number of row groups: {parquet_file.num_row_groups}')
    for i in range(parquet_file.num_row_groups):
        row_group = parquet_file.metadata.row_group(i)
        print(f'Row group {i}: {row_group.total_byte_size} bytes')

# Define the output directory for Parquet files
output_dir = '/mnt/sda4/parquet_files'
os.makedirs(output_dir, exist_ok=True)

# Define the columns for the TPC-H tables
tables = {
    'lineitem.tbl': [
        "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
        "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
        "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode",
        "l_comment"
    ],
    'orders.tbl': [
        "o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
        "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"
    ],
    'customer.tbl': [
        "c_custkey", "c_name", "c_address", "c_nationkey", "c_phone",
        "c_acctbal", "c_mktsegment", "c_comment"
    ],
    'supplier.tbl': [
        "s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone",
        "s_acctbal", "s_comment"
    ],
    'part.tbl': [
        "p_partkey", "p_name", "p_mfgr", "p_brand", "p_type",
        "p_size", "p_container", "p_retailprice", "p_comment"
    ],
    'partsupp.tbl': [
        "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"
    ],
    'nation.tbl': [
        "n_nationkey", "n_name", "n_regionkey", "n_comment"
    ],
    'region.tbl': [
        "r_regionkey", "r_name", "r_comment"
    ]
}

# Convert each .tbl file to Parquet
for file_name, columns in tables.items():
    tbl_file_path = file_name
    if os.path.exists(tbl_file_path):
        convert_to_parquet(tbl_file_path, columns, output_dir)
