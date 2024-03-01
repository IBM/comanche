#include <pistache/endpoint.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/buffer.h>
#include <parquet/api/reader.h>
#include <parquet/arrow/writer.h>
#include <arrow/io/memory.h>
#include <chrono>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h> // Include this header
#include "/home/ubuntu/json/include/nlohmann/json.hpp"
#include "SQLParser.h"
#include <fstream>



using json = nlohmann::json;


std::vector<uint8_t> FetchParquetMetadata(const Aws::S3::S3Client& s3Client, const std::string& bucket, const std::string& key) {
    // Fetch content length using HeadObject
    Aws::S3::Model::HeadObjectRequest headObjectRequest;
    headObjectRequest.WithBucket(bucket.c_str()).WithKey(key.c_str());
    auto headObjectOutcome = s3Client.HeadObject(headObjectRequest);

    if (!headObjectOutcome.IsSuccess()) {
        throw std::runtime_error("Failed to fetch object head for " + key);
    }

    auto contentLength = headObjectOutcome.GetResult().GetContentLength();

    // Fetch the last 8 bytes to get metadata size
    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.WithBucket(bucket.c_str()).WithKey(key.c_str());
    getObjectRequest.SetRange("bytes=" + std::to_string(contentLength - 8) + "-" + std::to_string(contentLength));
    auto getObjectOutcome = s3Client.GetObject(getObjectRequest);

    if (!getObjectOutcome.IsSuccess()) {
        throw std::runtime_error("Failed to fetch object footer for " + key);
    }

    Aws::IOStream& responseStream = getObjectOutcome.GetResult().GetBody();
    std::string endContent(std::istreambuf_iterator<char>(responseStream), {});

    // Check magic number
    if (endContent.substr(4) != "PAR1") {
        throw std::runtime_error("File does not look like a Parquet file; magic number incorrect");
    }

    int32_t fileMetaLength;
    memcpy(&fileMetaLength, endContent.data(), sizeof(fileMetaLength));

    // Fetch the file metadata
    getObjectRequest.SetRange("bytes=" + std::to_string(contentLength - 8 - fileMetaLength) + "-" + std::to_string(contentLength - 8));
    getObjectOutcome = s3Client.GetObject(getObjectRequest);

    if (!getObjectOutcome.IsSuccess()) {
        throw std::runtime_error("Failed to fetch object metadata for " + key);
    }

    Aws::IOStream& metaStream = getObjectOutcome.GetResult().GetBody();
    


// Determine the size of the stream
metaStream.seekg(0, std::ios::end);
size_t size = metaStream.tellg();
metaStream.seekg(0, std::ios::beg);

// Read directly into a std::vector<uint8_t>
std::vector<uint8_t> bufferData(size);
metaStream.read(reinterpret_cast<char*>(bufferData.data()), size);







    return bufferData;
}



std::vector<uint8_t> ReadRowGroupFromS3(const Aws::S3::S3Client& s3Client, const std::string& bucket, const std::string& key, const std::shared_ptr<parquet::FileMetaData>& metadata, int rowGroupIndex) {
    if (rowGroupIndex >= metadata->num_row_groups()) {
        throw std::runtime_error("Row group index out of bounds.");
    }

    const auto rowGroup = metadata->RowGroup(rowGroupIndex);
    const int64_t group_offset = rowGroup->file_offset();
    const int64_t group_size = rowGroup->total_byte_size();

    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.SetBucket(bucket.c_str());
    getObjectRequest.SetKey(key.c_str());

    // Specify the byte range to fetch
    std::string range = "bytes=" + std::to_string(group_offset) + "-" + std::to_string(group_offset + group_size - 1);
    getObjectRequest.SetRange(range.c_str());

    auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
    if (!getObjectOutcome.IsSuccess()) {
        std::cerr << "Failed to fetch row group from S3: " << getObjectOutcome.GetError().GetMessage() << std::endl;
        throw std::runtime_error("Failed to fetch row group data from S3.");
    }

    Aws::IOStream& objectStream = getObjectOutcome.GetResult().GetBody();
   std::string temp(std::istreambuf_iterator<char>(objectStream), {});
    std::vector<uint8_t> rowGroupData(temp.begin(), temp.end());

    return rowGroupData;
}

void PrintParquetMetadata(const std::shared_ptr<parquet::FileMetaData>& metadata) {
    if (!metadata) {
        std::cerr << "Metadata is null." << std::endl;
        return;
    }

    // Print basic file metadata
    std::cout << "Version: " << metadata->version() << std::endl;
    std::cout << "Number of rows: " << metadata->num_rows() << std::endl;
    std::cout << "Number of row groups: " << metadata->num_row_groups() << std::endl;
    std::cout << "Created by: " << metadata->created_by() << std::endl;

    // Print schema details
    auto schema = metadata->schema();
    std::cout << "Schema: " << std::endl;
    for (int i = 0; i < schema->num_columns(); ++i) {
        auto column = schema->Column(i);
        std::cout << "  Column " << i << ": " << column->name()
                  << ", Type: " << column->physical_type()
                 //<< ", Repetition: " << column->repetition_type()
                  << std::endl;
    }

    // Print row group details
    for (int rg = 0; rg < metadata->num_row_groups(); ++rg) {
        auto row_group = metadata->RowGroup(rg);
        std::cout << "Row Group " << rg
                  << ", Total byte size: " << row_group->total_byte_size()
                  << ", Number of rows: " << row_group->num_rows()
                  << std::endl;

        // Print column chunk details within each row group
       /* for (int col = 0; col < row_group->num_columns(); ++col) {
            auto column_chunk = row_group->ColumnChunk(col);
            std::cout << "  Column " << col
                      << ", Compressed size: " << column_chunk->total_compressed_size()
                      << ", Uncompressed size: " << column_chunk->total_uncompressed_size()
                      << std::endl;
        }*/
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

        // Example for measuring S3 fetch time
        auto start_s3_fetch = std::chrono::high_resolution_clock::now();


        // MinIO server connection parameters
        Aws::String minioEndpointUrl = "https://10.10.10.18:9000";
        Aws::String awsAccessKey = "minioadmin";
        Aws::String awsSecretKey = "minioadmin";

        // Create S3 client configuration
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.endpointOverride = minioEndpointUrl;
        clientConfig.scheme = Aws::Http::Scheme::HTTPS;
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


////////////////////////////////////////////////////
//Read parquet metadata

            auto bufferData = FetchParquetMetadata(s3Client, bucket, key);


            // Create an arrow::Buffer from the std::vector<uint8_t>
            auto buffer = std::make_shared<arrow::Buffer>(bufferData.data(), bufferData.size());

            // Convert the size to uint32_t for the Make function
            uint32_t metadataSize = static_cast<uint32_t>(buffer->size());

            std::shared_ptr<parquet::FileMetaData> metadata;
            try {
                // Correctly pass a pointer to the metadata size
                metadata = parquet::FileMetaData::Make(buffer->data(), &metadataSize);
            } catch (const std::exception& e) {
                // Handle any errors that might occur during the parsing of the metadata
                throw std::runtime_error(std::string("Failed to parse Parquet metadata: ") + e.what());
            }



            if (metadata) { // Check if metadata is not null
                std::cout << "Number of row groups: " << metadata->num_row_groups() << std::endl;
            } else {
                std::cerr << "Failed to retrieve metadata." << std::endl;
            }

        
////////////////////////////////////////////////////       
//Determine which row group indices to check
//std::vector<std::pair<int64_t, int64_t>> requiredRowGroups = determineRequiredRowGroups(fileMetadata, sqlExpression);

////////////////////////////////////////////////////
            for (int x = 0; x < metadata->num_row_groups(); x++) {
                try {
                    // Call the function to read the row group from S3
                    std::vector<uint8_t> rowGroupData = ReadRowGroupFromS3(s3Client, bucket, key, metadata, x);
                    //std::cout << "Row group " << x << " data size: " << rowGroupData.size() << " bytes" << std::endl;

                    // Find the index of the column with ID "ID"
                    int column_index = -1;

                    for (int i = 0; i < metadata->schema()->num_columns(); i++) {
                        if (metadata->schema()->Column(i)->name() == "ID") {
                           //std::cout << metadata->schema()->Column(i)->name() << std::endl;
                            column_index = i;
                            break;
                        }
                    }

                    if (column_index == -1) {
                        std::cerr << "Column 'ID' not found." << std::endl;
                        continue; // Skip to the next row group if the column is not found
                    }
                     auto row_group = metadata->RowGroup(x);
                            
                    const auto& column_chunk = row_group->ColumnChunk(column_index);
                    int64_t column_offset = column_chunk->file_offset();
                    int64_t column_size = column_chunk->total_compressed_size(); // Or use total_uncompressed_size() based on your need

                    //std::cout << "Row group " << x << ", Column 'ID' offset: " << column_offset << ", size: " << column_size << " bytes" << std::endl;


                    auto buffer = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(rowGroupData.data()), rowGroupData.size());
                    auto stream = std::make_shared<arrow::io::BufferReader>(buffer);


                }
                catch (const std::exception& e) {
                    std::cerr << "Failed to read row group " << x << ": " << e.what() << std::endl;

                }
            }

            auto rowGroupA = ReadRowGroupFromS3(s3Client, bucket, key, metadata, 0);

      

            PrintParquetMetadata(metadata);
////////////////////////////////////////////////////

            auto start_stream = std::chrono::high_resolution_clock::now();
            auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
            if (getObjectOutcome.IsSuccess()) {
                // Read the Parquet data from S3


            auto& objectStream = getObjectOutcome.GetResult().GetBody();


            auto end_s3_fetch = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> s3_fetch_duration = end_s3_fetch - start_s3_fetch;
            std::cout << "Time to fetch object from S3: " << s3_fetch_duration.count() << " seconds" << std::endl;


/////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////

                
                std::vector<char> data;
                
                objectStream.seekg(0, std::ios::end);
                std::streamsize size = objectStream.tellg();
                objectStream.seekg(0, std::ios::beg);



		        if (size > 0) {
    		        data.resize(static_cast<size_t>(size)); // Resize the vector to the exact size of the stream
    	            objectStream.read(data.data(), size); // Read the entire stream at once
		        }

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

                auto start_table = std::chrono::high_resolution_clock::now();
                // Read entire file as a single Arrow table
                std::shared_ptr<arrow::Table> table;
                status = arrowReader->ReadTable(&table);
  

                auto end_table = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> table_duration = end_table - start_table;
                std::cout << "Time to read table: " << buffer_duration.count() << " seconds" << std::endl;  
      
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

                auto start_filter = std::chrono::high_resolution_clock::now();

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

                auto end_filter = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> filter_duration = end_filter - start_filter;
                std::cout << "Time to wrap in dataset and filter: " << filter_duration.count() << " seconds" << std::endl; 
///////////////////////////////////////////////////////////////////

                // Print the Arrow table's schema
                //std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;
                // Print the Arrow table's data
                //std::cout << "Table Data:\n" << result_table.ValueUnsafe()->ToString() << std::endl;


                

         
                //response.send(Http::Code::Ok, "jo");
            } else {
                // Error handling
                std::cerr << "Failed to get object: " << getObjectOutcome.GetError().GetMessage() << std::endl;
            }
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


