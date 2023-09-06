#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <program_name> <filter_id1:parquet_filename1> [<filter_id2:parquet_filename2> ...]"
    exit 1
fi

program_name=$1

# Loop through the provided filter IDs and Parquet filenames
shift 1 # Remove program name from arguments
while [ "$#" -gt 0 ]; do
    arg=$1
    filter_id=${arg%%:*}
    parquet_filename=${arg#*:}
    
    echo "Running program for Filter ID: $filter_id, Parquet file: $parquet_filename"
    
    # Measure the execution time using the 'time' command
    /usr/bin/time -f "Execution time: %E" ./$program_name "$parquet_filename" "$filter_id"
    
    echo "-----------------------------------"
    
    shift 1 # Move to the next argument
done
