import requests

url = "https://10.10.10.20/intercepted_s3_get"
payload = {
    "bucket": "mycsvbucket",
    "key": "sampledata/dataStat_10000.parquet",
    "sql": "SELECT * FROM s3object WHERE ID < 120 and Age > 60"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers, verify=False)
print(response.text)

