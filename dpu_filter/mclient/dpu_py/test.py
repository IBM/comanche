import boto3
from flask import Flask, request
import io
import pyarrow as pa
import pyarrow.parquet as pq
from fastparquet import thrift_structures as TS
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
    verify=False)

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


def read_row_group_from_s3(bucket, key, group_offset, group_size):
    # Calculate the range for fetching the row group data
    range_header = f"bytes={group_offset}-{group_offset + group_size - 1}"
    print(group_offset)
    print(group_size)
    # Fetch the row group data from S3

    response = s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
    row_group_data = response['Body'].read()

    # Convert the binary data to a file-like object
    row_group = io.BytesIO(row_group_data)

    return row_group


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

        schema = pa.schema([pa.field(col['name'], pa.type_for_alias(col['type'])) for col in metadata['columns']])
        row_groups = []

        id_index = None
        age_index = None

        # Iterate through all column names in the schema
        for i, schema_field in enumerate(schema):
            column_name = schema_field.name

            # Check if the column name matches "ID" or "Age"
            if column_name == "ID":
                id_index = i
            elif column_name == "Age":
                age_index = i

        # Row groups to be fetched
        group_indices = []

        id_index = id_index - 1
        age_index = age_index - 1
        for i, row_group_meta in enumerate(metadata['row_groups']):
            id_column_stats = row_group_meta['columns'][id_index]['statistics']
            age_column_stats = row_group_meta['columns'][age_index]['statistics']

            # Check if ID falls within range
            if id_column_stats['min'] <= id <= id_column_stats['max']:

                if age_column_stats['min'] <= age <= age_column_stats['max']:
                    group_indices.append(i)

        print("Row groups that match: ", group_indices)

        # Create an empty Parquet file with the schema
        parquet_writer = pq.ParquetWriter('tmp.parquet', schema)

        for j in group_indices:
            group_offset = metadata['row_groups'][j]['start']
            group_size = metadata['row_groups'][j]['end'] - group_offset
            row_group_data = read_row_group_from_s3(bucket, key, group_offset, group_size)

            # Add row group to the Parquet file
            parquet_writer.write_table(pa.ipc.open_stream(row_group_data))

        # Close the Parquet writer
        parquet_writer.close()

        # Read the Parquet file as a byte stream
        with open('tmp.parquet', 'rb') as f:
            parquet_data = f.read()

        # Return the Parquet data
        return parquet_data, 200, {'Content-Type': 'application/octet-stream'}

    except Exception as e:
        return str(e), 500  # Handle exceptions gracefully

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

