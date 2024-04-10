#include <pistache/endpoint.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <chrono>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h> // Include this header
#include "/home/ubuntu/json/include/nlohmann/json.hpp"
#include "SQLParser.h"



using json = nlohmann::json;

// Callback function
void GetObjectCallback(const Aws::S3::S3Client* client,
                       const Aws::S3::Model::GetObjectRequest& request,
                       const Aws::S3::Model::GetObjectOutcome& outcome,
                       const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) {
    // Check if the request was successful
    if (outcome.IsSuccess()) {
        // Success
        std::cout << "Successfully downloaded object." << std::endl;
         auto& stream = outcome.GetResult().GetBody();
        // Process the outcome as needed
    } else {
        // Failure
        std::cerr << "Failed to download object: " << outcome.GetError().GetMessage() << std::endl;
    }
}

int main() {

    if (setenv("AWS_EC2_METADATA_DISABLED", "true", 1) != 0) {
        // Handle error if needed
    }

        
        auto start_time = std::chrono::high_resolution_clock::now();

            // Extract bucket, key, and SQL expression
        std::string bucket = "mycsvbucket" ;
        std::string key = "sampledata/dataStat_1000000.parquet";
        std::string sqlExpression = "SELECT * FROM s3object WHERE ID < 120";

        // Initialize AWS SDK
        Aws::SDKOptions options;
        Aws::InitAPI(options);




        // MinIO server connection parameters
        Aws::String minioEndpointUrl = "http://10.10.10.18:9000";
        Aws::String awsAccessKey = "minioadmin";
        Aws::String awsSecretKey = "minioadmin";

        // Create S3 client configuration
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.endpointOverride = minioEndpointUrl;
        clientConfig.scheme = Aws::Http::Scheme::HTTP;//HTTPS;
        clientConfig.verifySSL = false;

        // Create AWSCredentials object
        Aws::Auth::AWSCredentials credentials(awsAccessKey, awsSecretKey);

        // Create S3 client
        Aws::S3::S3Client s3Client(credentials, clientConfig,
                                    Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);




        try {
            // Get the object from S3
            Aws::S3::Model::GetObjectRequest getObjectRequest;
            getObjectRequest.SetBucket(bucket.c_str());
            getObjectRequest.SetKey(key.c_str());


            // Example for measuring S3 fetch time
            auto start_s3_fetch = std::chrono::high_resolution_clock::now();

             s3Client.GetObjectAsync(getObjectRequest, GetObjectCallback, nullptr);

                            auto end_s3_fetch = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> s3_fetch_duration = end_s3_fetch - start_s3_fetch;
                std::cout << "Time to fetch object from S3: " << s3_fetch_duration.count() << " seconds" << std::endl;
            
    

        } catch (const std::exception& e) {
            // Exception handling
            std::cerr << "Exception: " << e.what() << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_duration = end_time - start_time;
        std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl; 

        // Shutdown AWS SDK
        Aws::ShutdownAPI(options);

    return 0;
}


/*
// Get the number of row groups in the file
int num_row_groups = arrowReader->num_row_groups();

// Process each row group individually
for (int row_group_index = 0; row_group_index < num_row_groups; ++row_group_index) {
    // Read data from the current row group
    std::shared_ptr<arrow::Table> table;
    status = arrowReader->RowGroup(row_group_index)->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading row group " << row_group_index << ": " << status.message() << std::endl;
        continue; // Skip to the next row group
    }

    // Apply SQL filter expression to the table (if needed)
    // Your existing filter logic here...

    // Print the Arrow table's data
    std::cout << "Table Data (Row Group " << row_group_index << "):\n" << table->ToString() << std::endl;
}
*/
