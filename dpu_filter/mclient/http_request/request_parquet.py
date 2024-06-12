import requests
import time
import pyarrow.parquet as pq
import pyarrow as pa
import os

start_time = time.time()
url = "http://10.10.10.20:8080/data"
payload = {
    "bucket": "mycsvbucket",
    "key": "sampledata/dataStat_1000000.parquet",
    "sql": "SELECT * FROM s3object WHERE Age > 60",
    # "sql": "SELECT * FROM s3object WHERE ID < 120 and Age > 60",
    # "sql": "SELECT * FROM s3object WHERE ID < 120"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers, verify=False)
end_time = time.time()

if response.status_code == 200:
    output_filename = 'filtered_output.parquet'
    with open(output_filename, 'wb') as f:
        f.write(response.content)
    print(f"Success: Parquet file saved as {output_filename}")
    
    # Read and print the Parquet file content
    table = pq.read_table(output_filename)
    print(table.to_pandas())  # Print as a Pandas DataFrame
else:
    print("HTTP Error:", response.status_code)

print("Time taken:", end_time - start_time, "seconds")

