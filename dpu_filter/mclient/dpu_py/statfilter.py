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
       
        # Fetch Parquet file metadata
        metadata = pq.ParquetFile(parquet_data).metadata


        print(metadata.schema)


        # Print min and max statistics for the 'ID' column in each row group
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)

            print(f"Row Group {row_group_index} Statistics:")
    
            # Access column chunks in the row group
            for column_index in range(row_group.num_columns):
                column_chunk = row_group.column(column_index)

                # Access statistics for the column chunk (assuming 'ID' is the column name)
                if column_chunk.name == 'ID':
                    print("Column 'ID' Statistics:")
                    print("Min:", column_chunk.statistics.min)
                    print("Max:", column_chunk.statistics.max)
      
        # Fetch Parquet file metadata
        metadata = pq.ParquetFile(parquet_data).metadata
        print(metadata)
        # Get the schema of the Parquet file
        schema = metadata.schema

        # Print column information
        print("Columns:")
        for i, column in enumerate(schema):
            print(f"Column {i+1}: {column.name}")
            print(f"  Min: {column.min()}")
            print(f"  Max: {column.max()}")
        # Print row groups and min-max values
        print("Row groups:")
        for i, row_group in enumerate(metadata.row_groups):
            print(f"Row group {i+1}:")
            for column in row_group.columns:
                print(f"Column {column.path_in_schema}: Min={column.statistics.min}, Max={column.statistics.max}")
         
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

        # Convert the filtered DataFrame to JSON
        json_response = result.to_json(orient='records')

        return json_response, 200

    except Exception as e:
        return str(e), 500  # Handle exceptions gracefully

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

