import boto3
import io
import pyarrow.parquet as pq
from flask import Flask, request, jsonify
import sqlparse
import re

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

# Endpoint to handle intercepted S3 get_object requests
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

        # Parse the SQL expression to extract column name and value for filtering
        parsed_sql = sqlparse.parse(sql_expression)
        print("Parsed SQL:", parsed_sql)
        if len(parsed_sql) != 1 or not isinstance(parsed_sql[0], sqlparse.sql.Statement):
            raise ValueError("Invalid SQL expression format")

        stmt = parsed_sql[0]
        print("SQL Statement tokens:", stmt.tokens)

        # Extract the WHERE clause
        where_clause = None
        for token in stmt.tokens:
            if isinstance(token, sqlparse.sql.Where):
                where_clause = token
                break

        if where_clause is None:
            raise ValueError("WHERE clause not found in SQL expression")

        # Extract the comparison operator and value using regular expressions
        match = re.match(r'.*?(\S+)\s*([<>!=]+)\s*(\S+)', where_clause.value)
        if match is None:
            raise ValueError("Invalid comparison format in WHERE clause")

        column_name, comparison_operator, value_str = match.groups()
        value = int(value_str)

        # Filter the DataFrame based on the extracted column name, comparison operator, and value
        if comparison_operator == '<':
            filtered_df = df[df[column_name] < value]
        elif comparison_operator == '>':
            filtered_df = df[df[column_name] > value]
        elif comparison_operator == '=':
            filtered_df = df[df[column_name] == value]
        # Add support for other comparison operators as needed

        # Convert the filtered DataFrame to JSON
        json_response = filtered_df.to_json(orient='records')

        return json_response, 200

    except Exception as e:
        return str(e), 500  # Handle exceptions gracefully

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

