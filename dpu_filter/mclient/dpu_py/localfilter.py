import boto3
import io
import pyarrow.parquet as pq
from flask import Flask, request, jsonify
import sqlparse
import time

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
    
        # Time measurement for full execution
        start_time = time.time()
        
        # Time measurement for download
        download_start_time = time.time()
        
        # Fetch the Parquet file from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)

        # Read the Parquet data into memory
        parquet_data = io.BytesIO(response['Body'].read())
        

        # Calculate the download time
        download_end_time = time.time()
        download_time = download_end_time - download_start_time
        print("Download time:", download_time, "seconds")
 
        conversion_start_time = time.time()
        # Read the Parquet data as a Pandas DataFrame
        df = pq.read_table(parquet_data).to_pandas()

        conversion_end_time = time.time()
        conversion_time = conversion_end_time - conversion_start_time
        print("Conversion time:", conversion_time, "seconds")

        sql_start_time = time.time()
        # If no SQL expression is provided, return the entire DataFrame
        if not sql_expression:
            json_response = df.to_json(orient='records')
            return json_response, 200

        # Parse the SQL expression to extract the SELECT and WHERE clauses
        parsed_sql = sqlparse.parse(sql_expression)[0]

        # Initialize selected_column to None
        selected_column = None

        # Check if the SELECT clause is not wildcard
        if parsed_sql.tokens[2].value != '*':
            # Extract the column name from the SELECT clause token
            selected_column = parsed_sql.tokens[2]

        # Initialize where_clause_value to an empty string
        where_clause_value = ''

        # Look for the WHERE clause in the parsed SQL tokens
        for token in parsed_sql.tokens:
            if isinstance(token, sqlparse.sql.Where):
                # Extract the WHERE clause value and remove the word "WHERE"
                where_clause_value = token.value.replace("WHERE", "", 1).strip()
                break
 
        sql_end_time = time.time()

        sql_time = sql_end_time - sql_start_time
        print("SQL parse time:", sql_time, "seconds")

        filter_start_time = time.time()
        # If no WHERE clause is found, set result to the entire DataFrame
        if not where_clause_value:
            result = df
        else:
            # Filter the DataFrame based on the WHERE clause value
            result = df.query(where_clause_value)

        # If a specific column is selected, filter the DataFrame to include only that column
        if selected_column is not None:
            selected_column = selected_column.get_name()
            result = result[[selected_column]]


        filter_end_time = time.time()
        filter_time = filter_end_time - filter_start_time
        print("Filter time:", filter_time, "seconds")

        # Convert the filtered DataFrame to JSON
        json_response = result.to_json(orient='records')
        
        end_time = time.time()
        full_execution_time = end_time - start_time
        print("Full execution time:", full_execution_time, "seconds")
        return json_response, 200

    except Exception as e:
        return str(e), 500  # Handle exceptions gracefully

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

