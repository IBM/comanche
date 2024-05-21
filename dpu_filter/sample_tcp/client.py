import requests
import time

server_url = "https://10.10.10.18/dataStat_1000000.parquet"

# Start timing
start_time = time.time()
response = requests.get(server_url, verify=False)
# End timing
end_time = time.time()

# Calculate elapsed time
#elapsed_time = (end_time - start_time)
content_size = len(response.content)

download_path = "/home/ubuntu/test.parquet"
with open(download_path, 'wb') as file:
    file.write(response.content)

# End timing
#end_time = time.time()

# Calculate elapsed time
elapsed_time = (end_time - start_time)

print(f"File downloaded successfully in {elapsed_time: .3f} seconds.")
print(f"Total bytes downloaded: {content_size} bytes.")
