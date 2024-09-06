import requests
import time
import pyarrow.parquet as pq
import pyarrow as pa
import os
import json

# Function to process Parquet output
def process_parquet(content, output_filename):
    with open(output_filename, 'wb') as f:
        f.write(content)
    print(f"Success: Parquet file saved as {output_filename}")
    
    # Read and print the Parquet file content
    table = pq.read_table(output_filename)
    df = table.to_pandas()  # Convert to a Pandas DataFrame
    print(df)  # Print the DataFrame
    print(f"Number of rows: {len(df)}")  # Print the number of rows

# Function to process JSON output
def process_json(content):
    try:
        data = json.loads(content)
        print("JSON output:")
        print(json.dumps(data, indent=2))  # Pretty print JSON
        
        # Check if "table" exists in the JSON data
        if "table" in data:
            table_content = data["table"]
            lines = table_content.splitlines()
            num_rows = len(lines) - 1  # Subtract 1 for the header row
            print(f"Number of rows: {num_rows}")
        else:
            print("The 'table' field is missing from the JSON output.")
            # You can add additional handling here if needed
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Raw content:")
        print(content)


start_time = time.time()
url = "http://10.10.10.20:8080/data"
output_format = "json"  # Change to "parquet" for Parquet output

payload = {
    "bucket": "encrypted",
    "key": "enc_web_sales.parquet",
    "sql": "SELECT ws_item_sk, ws_quantity, ws_sales_price FROM S3Object WHERE ws_sold_date_sk BETWEEN 2451911 AND 2452640", # 10%
    "output": output_format
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers, verify=False)
end_time = time.time()

print("Time taken:", end_time - start_time, "seconds")

if response.status_code == 200:
    if output_format == "parquet":
        output_filename = 'filtered_output.parquet'
        process_parquet(response.content, output_filename)
    elif output_format == "json":
        process_json(response.content)
else:
    print("HTTP Error:", response.status_code)

