import boto3
from flask import Flask, request, jsonify
from fastparquet.cencoding import from_buffer
import pyarrow.parquet as pq
import io
import pandas as pd
import pyarrow as pa
import codecs
import pyarrow.ipc as ipc


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




import pyarrow as pa

def parquet_metadata_to_arrow_schema(metadata):
    fields = []
    for schema_field in metadata.schema:
        print(schema_field)
        name = schema_field.name.decode('utf-8')
        
        print(name)

        arrow_type = schema_field.type
        print(arrow_type)
        # Create Arrow field based on Parquet schema
        arrow_field = pa.field(name, parquet_type)
        fields.append(arrow_field)
    return pa.schema(fields)


def read_row_group_from_s3(bucket, key, group_offset, group_size):
    # Calculate the range for fetching the row group data
    range_header = f"bytes={group_offset}-{group_offset + group_size - 1}"
    print(group_offset)
    print(group_size)
    # Fetch the row group data from S3
    
    response = s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
    row_group_data = response['Body'].read()


    # Convert the binary data to a file-like object
    #row_group = io.BytesIO(row_group_data)

    row_group_buffer = pa.py_buffer(row_group_data)
    row_group_buffer_reader = pa.BufferReader(row_group_buffer)

    return row_group_buffer_reader



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


        age = 60
        id = 120

        schema = metadata.schema
        print(schema)
        #arrow_schema = parquet_metadata_to_arrow_schema(metadata)
        #arrow_schema = create_arrow_schema(schema)

        #print(arrow_schema)
        #schema_field = metadata.schema[2]
        #name = schema_field.name.decode('utf-8')
        # Print the column name
        #print(name)
       
        id_index = None
        age_index = None

        # Iterate through all column names in the schema
        for i, schema_field in enumerate(schema):
        # Decode the column name from bytes to string
            column_name = schema_field.name.decode('utf-8')
            print(column_name)
            arrow_type = schema_field.type
            print(arrow_type)
                         
            # Check if the column name matches "ID" or "Age"
            if column_name == "ID":
                id_index = i   
            elif column_name == "Age":
                age_index = i

        #Need to change it so that it works with every type, maybe get type from schema
        #min_value_int = int.from_bytes(metadata.row_groups[0].columns[id_index].meta_data.statistics.min_value, byteorder='little')
        #max_value_int = int.from_bytes(metadata.row_groups[0].columns[1].meta_data.statistics.max_value, byteorder='little')

        # Print the integer values
        #print("Min value:", min_value_int)
        #print("Max value:", max_value_int)

        #Row groups to be fetched
        group_indices = []

        id_index = id_index - 1
        age_index = age_index -1 
        for i, row_group in enumerate(metadata.row_groups):
            id_column_stats = row_group.columns[id_index].meta_data.statistics
            age_column_stats = row_group.columns[age_index].meta_data.statistics
        
            #Check if ID falls within range
            if int.from_bytes(id_column_stats.min_value,byteorder='little') <= id <= int.from_bytes(id_column_stats.max_value,byteorder='little'):
                #print("Min value:", int.from_bytes(id_column_stats.min_value,byteorder='little'))
                #print("Max value:", int.from_bytes(id_column_stats.max_value,byteorder='little'))
               
                #print("Age Min value:", int.from_bytes(age_column_stats.min_value,byteorder='little'))
                #print("Age Max value:", int.from_bytes(age_column_stats.max_value,byteorder='little'))

                if int.from_bytes(age_column_stats.min_value,byteorder='little') <= age <= int.from_bytes(age_column_stats.max_value,byteorder='little'):
                    group_indices.append(i)
        
    
        print("Row groups that match: ", group_indices)

        for j in group_indices:
            group_offset = metadata.row_groups[j].file_offset
            group_size = metadata.row_groups[j].total_byte_size
            row_group_data = read_row_group_from_s3(bucket, key, group_offset, group_size)
            ddg = convert_row_group_to_dataframe(row_group_buffer)
            

            print("h")
           
            #Add row group to parquet file created
        metadata_str=str(metadata)
        return metadata_str, 200
    except Exception as e:
        return str(e), 500  # Handle exceptions gracefully

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
