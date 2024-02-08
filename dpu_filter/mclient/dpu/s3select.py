import boto3
from flask import Flask, request

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
    verify=False
)

# Endpoint to handle intercepted S3 select_object_content requests
@app.route('/intercepted_s3_get', methods=['POST'])
def intercepted_s3_select():
    data = request.get_json()
    bucket = data.get('bucket')
    key = data.get('key')
    sql_expression = data.get('sql')

    # Print the intercepted bucket, key, and SQL expression
    print(f'Intercepted: Bucket={bucket}, Key={key}, SQL={sql_expression}')

    try:
        response = s3_client.select_object_content(
            Bucket=bucket,
            Key=key,
            ExpressionType='SQL',
            Expression=sql_expression,
            InputSerialization={'Parquet': {}},
            OutputSerialization={'JSON': {}}
        )

        # Process response payload
        for event in response['Payload']:
            if 'Records' in event:
                records = event['Records']['Payload'].decode('utf-8')
               # print(records)
                return records, 200
        

        return 'No records found', 4040
    except Exception as e:
        return str(e), 500  # Handle exceptions gracefully

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

