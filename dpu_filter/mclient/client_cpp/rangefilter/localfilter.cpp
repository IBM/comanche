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


using json = nlohmann::json;

// Function to download a chunk of the file
void download_chunk(const Aws::S3::S3Client& s3Client, const std::string& bucket, const std::string& key, long long start, long long end, std::vector<char>& buffer, std::mutex& buffer_mutex) {
    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.SetBucket(bucket.c_str());
    getObjectRequest.SetKey(key.c_str());
    std::string range = "bytes=" + std::to_string(start) + "-" + std::to_string(end);
    getObjectRequest.SetRange(range.c_str());

    auto getObjectOutcome = s3Client.GetObject(getObjectRequest);

    /*if (getObjectOutcome.IsSuccess()) {
        auto& objectData = getObjectOutcome.GetResult().GetBody();
        std::vector<char> chunkData(std::istreambuf_iterator<char>(objectData), {});

        {
            std::lock_guard<std::mutex> guard(buffer_mutex);
            std::copy(chunkData.begin(), chunkData.end(), buffer.begin() + start);
        }
    } else {
        std::cerr << "Failed to download chunk: " << getObjectOutcome.GetError().GetMessage() << std::endl;
    }*/
}

int main() {

    if (setenv("AWS_EC2_METADATA_DISABLED", "true", 1) != 0) {
        // Handle error if needed
    }

        const long long CHUNK_SIZE = 16 * 1024 * 1024; // 16 MB

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
  // Example for measuring S3 fetch time
           
            // Fetch content length using HeadObject
            // Assume bucket and key are initialized std::string objects
    Aws::S3::Model::HeadObjectRequest headObjectRequest;
    headObjectRequest.WithBucket(bucket.c_str()).WithKey(key.c_str());
    auto headObjectOutcome = s3Client.HeadObject(headObjectRequest);

    if (!headObjectOutcome.IsSuccess()) {
        throw std::runtime_error("Failed to fetch object head for " + key);
    }

    auto contentLength = headObjectOutcome.GetResult().GetContentLength();
        
            
 

            std::vector<char> data;//(contentLength);
            data.resize(static_cast<size_t>(contentLength));
            
            std::mutex buffer_mutex;


 auto start_s3_fetch = std::chrono::high_resolution_clock::now();
      /*     int num_chunks = contentLength / CHUNK_SIZE + (contentLength % CHUNK_SIZE ? 1 : 0);
 
            
          
        std::vector<std::future<void>> futures;
        auto start_s3_fetch = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_chunks; ++i) {
            long long start = i * CHUNK_SIZE;
            long long end = std::min(start + CHUNK_SIZE - 1, contentLength - 1);
            futures.emplace_back(std::async(std::launch::async, download_chunk, std::ref(s3Client), bucket, key, start, end, std::ref(data), std::ref(buffer_mutex)));
        }


        for (auto& fut : futures) {
            fut.get();
        }*/

                // Calculate ranges for two halves
            // Calculate the size of each part
        long long partSize = contentLength / 44;
        
        // Function to download a part
        auto download_part = [&](int part) {
            long long start = part * partSize;
            long long end = (part == 43) ? contentLength - 1 : (start + partSize - 1); // Last part goes to the end of the file
            download_chunk(s3Client, bucket, key, start, end, data, buffer_mutex);
        };

        // Start downloading in four threads
        std::thread threads[44];
        for (int i = 0; i < 44; ++i) {
            threads[i] = std::thread(download_part, i);
        }

        // Wait for all threads to complete
        for (int i = 0; i < 44; ++i) {
            threads[i].join();
        }


        auto end_s3_fetch = std::chrono::high_resolution_clock::now();

           

           
            std::chrono::duration<double> s3_fetch_duration = end_s3_fetch - start_s3_fetch;
            std::cout << "Time to fetch object from S3: " << s3_fetch_duration.count() << " seconds" << std::endl;


         
            

 ////////////////////////////////////////////////////////

                auto start_stream = std::chrono::high_resolution_clock::now();
               

/////////////////////// Faster way Estimate the size of the stream to reserve vector capacity upfront (optional)
             /*   std::vector<char> data;
                
                objectStream.seekg(0, std::ios::end);
                std::streamsize size = objectStream.tellg();
                objectStream.seekg(0, std::ios::beg);



		        if (size > 0) {
    		        data.resize(static_cast<size_t>(size)); // Resize the vector to the exact size of the stream
    	            objectStream.read(data.data(), size); // Read the entire stream at once
		        }*/

                auto end_stream = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> stream_duration = end_stream - start_stream;
                std::cout << "Time to read stream: " << stream_duration.count() << " seconds" << std::endl;
/////////////////////////////////////////////

                auto start_buffer = std::chrono::high_resolution_clock::now();
                // Create an Arrow buffer from the vector
                auto arrowBuffer = arrow::Buffer::Wrap(data.data(), data.size());
                // Create an Arrow BufferReader from the Arrow buffer
                auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);


                // Open Parquet file reader
                std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);
      
                auto end_buffer = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> buffer_duration = end_buffer - start_buffer;
                std::cout << "Time to preparse arrow buffer and arrowreader: " << buffer_duration.count() << " seconds" << std::endl;

////////////////////////////////////////////
   // Apply SQL filter expression to the table (if needed)
    // Your existing filter logic here...

////////////////////////////////////////////////////

                auto start_sql = std::chrono::high_resolution_clock::now();
                // Parse the SQL expression
                std::vector<Token> tokens = SQLParser::parse(sqlExpression);

                // Example: Analyze tokens and construct a filter (Basic and specific case handling)
                std::string columnName;
                std::string operatorSymbol;
                std::string value;

                for (const auto& token : tokens) {
                    if (token.type == TokenType::COLUMN) {
                        columnName = token.value;
                    } else if (token.type == TokenType::OPERATOR) {
                        operatorSymbol = token.value;
                    } else if (token.type == TokenType::LITERAL) {
                        value = token.value;
                    }
            // Extend with more complex logic as needed
                }

                    
                auto field_ref = arrow::compute::field_ref(columnName);
                int l_value = std::stoi(value);
                auto literal_value = arrow::compute::literal(l_value);

                arrow::compute::Expression filter_expression;

                // Build expression based on the operator
                if (operatorSymbol == "=") {
                    filter_expression = arrow::compute::equal(field_ref,literal_value);
                } else if (operatorSymbol == ">") {
                    filter_expression = arrow::compute::greater(field_ref,literal_value);
                } else if (operatorSymbol == ">=") {
                    filter_expression = arrow::compute::greater_equal(field_ref,literal_value);
                } else if (operatorSymbol == "<") {
                    filter_expression = arrow::compute::less(field_ref, literal_value);
                } else if (operatorSymbol == "<=") {
                    filter_expression = arrow::compute::less_equal(field_ref,literal_value);
                } else if (operatorSymbol == "!=") {
                    filter_expression = arrow::compute::not_equal(field_ref, literal_value);
                }

                auto end_sql = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> sql_duration = end_sql - start_sql;
                std::cout << "Time to sql parse and create filter expression: " << sql_duration.count() << " seconds" << std::endl; 

//////////////////////////////////////////////////////////////////


///////////

                auto start_table = std::chrono::high_resolution_clock::now();
                // Read entire file as a single Arrow table
                //std::shared_ptr<arrow::Table> table;
                //status = arrowReader->ReadTable(&table);
  
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

 
              
                // Wrap the Table in an InMemoryDataset
                std::shared_ptr<arrow::dataset::Dataset> dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

                // Build ScannerOptions for a Scanner to apply filter operation
                auto options = std::make_shared<arrow::dataset::ScanOptions>();

                // Build the Scanner
                auto builder = arrow::dataset::ScannerBuilder(dataset);     
                // Set the filter
                arrow::Status build_status = builder.Filter(filter_expression);

                auto scanner = builder.Finish();

                // Perform the Scan and retrieve filtered result as Table
                auto result_table = scanner.ValueOrDie()->ToTable();

               
                // Print the Arrow table's schema
                //std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;
                // Print the Arrow table's data
               // std::cout << "Table Data:\n" << result_table.ValueUnsafe()->ToString() << std::endl;


    
}

                auto end_table = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> table_duration = end_table - start_table;
                std::cout << "Time to read table: " << table_duration.count() << " seconds" << std::endl;  
                  

         
     
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
