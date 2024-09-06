import requests
import time
import pyarrow.parquet as pq
import pyarrow as pa
import os
import json

start_time = time.time()
url = "http://10.10.10.20:8080/data"
payload = {
    "bucket": "encrypted",
    "key": "enc_inventory.parquet",
    #"sql": "SELECT COUNT(*), SUM(inv_quantity_on_hand) FROM S3Object WHERE inv_item_sk BETWEEN 13 AND 299982 AND inv_date_sk BETWEEN 2451654 AND 2451714"
    #"sql": "SELECT COUNT(*), SUM(inv_quantity_on_hand) FROM S3Object WHERE inv_date_sk BETWEEN 2451576 AND 2451941"
    #"sql": "SELECT COUNT(*), SUM(inv_quantity_on_hand) FROM S3Object WHERE inv_item_sk BETWEEN 35446 AND 278356 AND inv_date_sk BETWEEN 2452070 AND 2452160"
    #"sql": "SELECT COUNT(*), SUM(inv_quantity_on_hand) FROM S3Object WHERE inv_date_sk BETWEEN 2451911 AND 2451969"
    #"sql": "SELECT COUNT(*), SUM(inv_quantity_on_hand) FROM S3Object WHERE inv_item_sk BETWEEN 3389 AND 296474 AND inv_date_sk BETWEEN 2450971 AND 2451031"
    #"sql":  "SELECT inv_item_sk inv_quantity_on_hand FROM S3Object WHERE inv_date_sk BETWEEN 2451179 AND 2451186" # 0.7%
    "sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2450997" #10%
    #"sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2451179" #20%
    #"sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2451361" #30%
    #"sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2451543" #40%
    #"sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2451725" #50%
    #"sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2451907" #60%
    #"sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2452089" #70%
    #"sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2452271" #80%
    #"sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2452453" #90%

}
headers = {"Content-Type": "application/json"}

#COUNT(*), SUM(inv_quantity_on_hand)
response = requests.post(url, json=payload, headers=headers, verify=False)
end_time = time.time()

#if response.status_code == 200:
#    print("Success:", response.text)
#    print("Bytes received:", len(response.content))
#else:
#    print("HTTP Error:", response.status_code)

print("Time taken:", end_time - start_time, "seconds")


if response.status_code == 200:
    output_filename = 'filtered_output.parquet'
    with open(output_filename, 'wb') as f:
        f.write(response.content)
    print(f"Success: Parquet file saved as {output_filename}")
    
    table = pq.read_table(output_filename)
    print(table.to_pandas())  # Print as a Pandas DataFrame
else:
    print("HTTP Error:", response.status_code)


