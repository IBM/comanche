import requests

url = "https://10.10.10.20/intercepted_s3_get"
payload = {
    "bucket": "mycsvbucket",
    "key": "sampledata/dataStat_10000.parquet",
    "sql": "select * from s3object s where s.ID < 120"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers, verify=False)
print(response.text)

