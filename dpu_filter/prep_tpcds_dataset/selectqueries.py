import pandas as pd
import os

# Load the Parquet file into a DataFrame
parquet_file_path = 'inventory.parquet'
df = pd.read_parquet(parquet_file_path)

total_rows = len(df)
date_column = 'inv_date_sk'

# Find the date ranges for 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, and 90% selectivity
quantiles = {
    0.10: df[date_column].quantile(0.10),
    0.20: df[date_column].quantile(0.20),
    0.30: df[date_column].quantile(0.30),
    0.40: df[date_column].quantile(0.40),
    0.50: df[date_column].quantile(0.50),
    0.60: df[date_column].quantile(0.60),
    0.70: df[date_column].quantile(0.70),
    0.80: df[date_column].quantile(0.80),
    0.90: df[date_column].quantile(0.90),
}

print("Date Ranges for Desired Selectivity Ratios:")
for ratio, date_value in quantiles.items():
    print(f"{int(ratio*100)}% Selectivity: {date_value}")

# Function to save DataFrame to a temporary Parquet file and measure its size
def measure_filtered_data_size(filtered_df, temp_file_path='temp_filtered.parquet'):
    filtered_df.to_parquet(temp_file_path)
    file_size = os.path.getsize(temp_file_path)
    os.remove(temp_file_path)
    return file_size

# List of adjusted queries with corresponding filtering conditions and expected selectivity ratios
adjusted_queries = [
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.10]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.10]),
        "expected_ratio": 0.10
    },
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.20]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.20]),
        "expected_ratio": 0.20
    },
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.30]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.30]),
        "expected_ratio": 0.30
    },
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.40]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.40]),
        "expected_ratio": 0.40
    },
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.50]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.50]),
        "expected_ratio": 0.50
    },
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.60]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.60]),
        "expected_ratio": 0.60
    },
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.70]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.70]),
        "expected_ratio": 0.70
    },
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.80]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.80]),
        "expected_ratio": 0.80
    },
    {
        "sql": f"SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= {quantiles[0.90]}",
        "columns": ["inv_item_sk", "inv_quantity_on_hand"],
        "filter": (None, quantiles[0.90]),
        "expected_ratio": 0.90
    }
]

# Process each adjusted query
for query in adjusted_queries:
    filtered_df = df[df['inv_date_sk'] <= query["filter"][1]]
    filtered_df = filtered_df[query["columns"]]  # Select the specified columns

    filtered_rows = len(filtered_df)
    select_ratio = filtered_rows / total_rows

    filtered_data_size = measure_filtered_data_size(filtered_df)

    print(f'Query: {query["sql"]}')
    print(f'Total Rows: {total_rows}')
    print(f'Filtered Rows: {filtered_rows}')
    print(f'Select Ratio: {select_ratio:.4f}')
    print(f'Filtered Data Size: {filtered_data_size} bytes')
    print(f'Expected Select Ratio: {query["expected_ratio"]:.4f}')
    print()