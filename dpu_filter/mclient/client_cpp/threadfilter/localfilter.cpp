#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <aws/s3/model/HeadObjectRequest.h>
#include <pistache/endpoint.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <vector>
#include <aws/s3/model/HeadObjectRequest.h>
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
#include <future>
#include <mutex>


void download_chunk(const Aws::S3::S3Client& s3Client, const std::string& bucket, const std::string& key, long long start, long long end, std::vector<char>& buffer, std::mutex& buffer_mutex) {
    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.SetBucket(bucket.c_str());
    getObjectRequest.SetKey(key.c_str());
    std::string range = "bytes=" + std::to_string(start) + "-" + std::to_string(end);
    getObjectRequest.SetRange(range.c_str());

    auto getObjectOutcome = s3Client.GetObject(getObjectRequest);

    if (getObjectOutcome.IsSuccess()) {
        auto& objectData = getObjectOutcome.GetResult().GetBody();
        std::vector<char> chunkData(std::istreambuf_iterator<char>(objectData), {});

        std::lock_guard<std::mutex> guard(buffer_mutex);
        std::copy(chunkData.begin(), chunkData.end(), buffer.begin() + start);
    } else {
        std::cerr << "Failed to download chunk: " << getObjectOutcome.GetError().GetMessage() << std::endl;
    }
}

int main() {


    if (setenv("AWS_EC2_METADATA_DISABLED", "true", 1) != 0) {
        // Handle error if needed
    }

  

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



        // Get object size
        Aws::S3::Model::HeadObjectRequest headObjectRequest;
        headObjectRequest.WithBucket(bucket).WithKey(key);
        auto headObjectOutcome = s3Client.HeadObject(headObjectRequest);
        if (!headObjectOutcome.IsSuccess()) {
            std::cerr << "Error: " << headObjectOutcome.GetError().GetMessage() << std::endl;
            return 1;
        }
        auto contentLength = headObjectOutcome.GetResult().GetContentLength();
        
        std::vector<char> data(contentLength);
        std::mutex buffer_mutex;

        // Calculate ranges for two halves
        long long firstHalfEnd = contentLength / 2;
        long long secondHalfStart = firstHalfEnd + 1;

        // Download function for threads
        auto download_range = [&](long long start, long long end) {
            download_chunk(s3Client, bucket, key, start, end, data, buffer_mutex);
        };

        // Start download in two threads
        std::thread firstHalf(download_range, 0, firstHalfEnd);
        std::thread secondHalf(download_range, secondHalfStart, contentLength - 1);

        // Wait for downloads to finish
        firstHalf.join();
        secondHalf.join();

        // At this point, `data` contains the downloaded content
        std::cout << "Download complete." << std::endl;
    }
    Aws::ShutdownAPI(options);
    return 0;
}
