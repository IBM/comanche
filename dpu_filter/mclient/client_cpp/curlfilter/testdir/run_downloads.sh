#!/bin/bash

# Array of files to download
FILES=(
    "http://10.10.10.18/dataStat_100000.parquet"
    "http://10.10.10.18/dataStat_500000.parquet"
    "http://10.10.10.18/dataStat_600000.parquet"
    "http://10.10.10.18/dataStat_1000000.parquet"
)

# Number of files to download in parallel (adjust as needed)
NUM_PROCESSES=${#FILES[@]}

for ((i=0; i<NUM_PROCESSES; i++)); do
    ./download_file ${FILES[i]} &
done

wait

