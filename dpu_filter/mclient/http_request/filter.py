import requests
import time

start_time = time.time()
url = "https://10.10.10.20/intercepted_s3_get"
payload = {
    "bucket": "mycsvbucket",
    "key": "sampledata/dataStat_100000.parquet",
   # "sql": "SELECT * FROM s3object WHERE Age > 60",
   # "sql": "SELECT * FROM s3object WHERE ID < 120 and Age > 60",
    "sql": "SELECT * FROM s3object WHERE ID < 120"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers, verify=False)
end_time = time.time()
print(response.text)
#end_time = time.time()

# Now 'result' contains the filtered DataFrame according to SQL expression
# You can perform further operations on 'result' as needed
# For example:
print("Time taken:", end_time - start_time, "seconds")
