import boto3
import urllib3
import pyarrow.parquet as pq
import io
import pandas as pd
import sqlparse
import time

# Disable SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

s3 = boto3.client('s3',
                  endpoint_url='https://10.10.10.18:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin',
                  verify=False)

# Time measurement for full execution
start_time = time.time()

# Time measurement for download
download_start_time = time.time()

response = s3.get_object(
    Bucket='mycsvbucket',
    Key='sampledata/dataStat_1000000.parquet'
)



parquet_data = io.BytesIO(response['Body'].read())
#read() used to retrieve the data from an object in S3

download_end_time = time.time()

# Calculate the download time
download_time = download_end_time - download_start_time
print("Download time:", download_time, "seconds")


# Time measurement for converting Parquet data to Pandas DataFrame
conversion_start_time = time.time()

df = pq.read_table(parquet_data).to_pandas()

conversion_end_time = time.time()

# Calculate the conversion time
conversion_time = conversion_end_time - conversion_start_time
print("Conversion time:", conversion_time, "seconds")

# Example SQL statement
sql_expression = "SELECT * FROM df WHERE ID < 120"

# Time measurement for SQL parsing
sql_start_time = time.time()
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

# Calculate the sql parse time
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

# Calculate the Parquet read time
filter_time = filter_end_time - filter_start_time
print("Filter time:", filter_time, "seconds")

# Now 'result' contains the filtered DataFrame according to SQL expression
# You can perform further operations on 'result' as needed
# For example:
print(result)

# Time measurement for full execution
end_time = time.time()
full_execution_time = end_time - start_time
print("Full execution time:", full_execution_time, "seconds")

