import boto3
import io
import pyarrow.parquet as pq
from flask import Flask, request, jsonify
import sqlparse

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
        # Fetch the Parquet file from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)

        # Read the Parquet data into memory
        parquet_data = io.BytesIO(response['Body'].read())

        # Read the Parquet data as a Pandas DataFrame
        df = pq.read_table(parquet_data).to_pandas()

        # Parse the SQL expression to extract the WHERE clause
        parsed_sql = sqlparse.parse(sql_expression)[0]

        where_clause = None
        for token in parsed_sql.tokens:
            print(token)
            if isinstance(token, sqlparse.sql.Where):
                where_clause = token
                break

        if where_clause is None:
            raise ValueError("WHERE clause not found in SQL expression")
        
        print(where_clause.value)
        # Extract the WHERE clause value and remove the word "WHERE"
        # Remove the word "WHERE" from the string
        where_clause_value = where_clause.value.replace("WHERE", "", 1).strip()
        print(where_clause_value)

        # Filter the DataFrame based on the WHERE clause value
        result = df.query(where_clause_value)
 
        # Convert the filtered DataFrame to JSON
        json_response = result.to_json(orient='records')

        return json_response, 200

    except Exception as e:
        return str(e), 500  # Handle exceptions gracefully

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
