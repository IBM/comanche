curl -k -X POST -i "https://10.10.10.20/intercepted_s3_get" -H "Content-Type: application/json" --data-binary '{"bucket":"mycsvbucket","key":"sampledata/dataStat_10000.parquet","sql":"SELECT * FROM S3Object WHERE ID <= 120"}' 
