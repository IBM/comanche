import boto3
from flask import Flask, request, jsonify
from fastparquet.cencoding import from_buffer

app = Flask(__name__)

# Configure the S3 client to interact with the Minio server on 10.10.10.18
minio_endpoint_url = 'https://10.10.10.18:9000'
aws_access_key = 'minioadmin'
aws_secret_key = 'minioadmin'

s3_client = boto3.client(
    's3',
    endpoint_url=minio_endpoint_url,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    verify=False )

# Function to fetch Parquet file metadata
def parquet_metadata_s3(bucket, key):
    metadata = s3_client.head_object(Bucket=bucket, Key=key)
    content_length = int(metadata['ContentLength'])

    end_response = s3_client.get_object(Bucket=bucket, Key=key, Range=f'bytes={content_length-8}-{content_length}')
    end_content = end_response['Body'].read()

    if end_content[-4:] != b'PAR1':
        raise ValueError(f'File at {key} does not look like a Parquet file; magic {end_content[-4:]}')

    file_meta_length = int.from_bytes(end_content[:4], byteorder='little')

    file_meta_response = s3_client.get_object(Bucket=bucket, Key=key,
                                              Range=f'bytes={content_length-8-file_meta_length}-{content_length-8}')
    file_meta_content = file_meta_response['Body'].read()

    fmd = from_buffer(file_meta_content, "FileMetaData")

    return fmd

# Endpoint to handle intercepted S3 GetObject requests
@app.route('/intercepted_s3_get', methods=['POST'])
def intercepted_s3_get():
    data = request.get_json()
    bucket = data.get('bucket')
    key = data.get('key')

        # Print the intercepted bucket and key
    print(f'Intercepted: Bucket={bucket}, Key={key}')

    try:
        metadata = parquet_metadata_s3(bucket, key)
        print(metadata)
        metadata_str=str(metadata)
        return metadata_str, 200
    except Exception as e:
        return str(e), 500  # Handle exceptions gracefully

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
