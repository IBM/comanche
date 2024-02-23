#include <pistache/endpoint.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/table.h>  
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/statistics.h>
#include <chrono>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h> // Include this header
#include "/home/ubuntu/json/include/nlohmann/json.hpp"
#include "SQLParser.h"

using namespace Pistache;
using json = nlohmann::json;


struct S3Handler : public Http::Handler {
    HTTP_PROTOTYPE(S3Handler)

    void onRequest(const Http::Request& req, Http::ResponseWriter response) override {


        json requestJson = json::parse(req.body());

            // Extract bucket, key, and SQL expression
        std::string bucket = requestJson["bucket"];
        std::string key = requestJson["key"];
        std::string sqlExpression = requestJson["sql"];

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

            auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
            if (getObjectOutcome.IsSuccess()) {
                // Read the Parquet data from S3
                auto& objectStream = getObjectOutcome.GetResult().GetBody();

                auto end_s3_fetch = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> s3_fetch_duration = end_s3_fetch - start_s3_fetch;
                std::cout << "Time to fetch object from S3: " << s3_fetch_duration.count() << " seconds" << std::endl; 

//////////////////////////////////////////////////////////

                auto start_stream = std::chrono::high_resolution_clock::now();

                // Read the stream into a vector: takes long time
                //std::vector<char> data(std::istreambuf_iterator<char>(objectStream), {});

/////////////////////// Faster way Estimate the size of the stream to reserve vector capacity upfront (optional)
                std::vector<char> data;
                
                objectStream.seekg(0, std::ios::end);
                std::streamsize size = objectStream.tellg();
                objectStream.seekg(0, std::ios::beg);

                if (size > 0) {
                    data.resize(static_cast<size_t>(size)); // Resize the vector to the exact size of the stream
                    objectStream.read(data.data(), size); // Read the entire stream at once
                }

                /*if (size > 0) {
                    data.reserve(static_cast<size_t>(size));
                }

                // Use a buffer to read chunks of the stream
                constexpr std::streamsize bufferSize = 8192*8; // Example buffer size, adjust as needed
                auto buffer = std::make_unique<char[]>(bufferSize); // Heap allocation with unique_ptr

                while (objectStream.read(buffer.get(), bufferSize) || objectStream.gcount() > 0) {
                    data.insert(data.end(), buffer.get(), buffer.get() + objectStream.gcount());
                }*/
    
                auto end_stream = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> stream_duration = end_stream - start_stream;
                std::cout << "Time to read stream: " << stream_duration.count() << " seconds" << std::endl;
///////////////////////////////////////////////
                auto start_buffer = std::chrono::high_resolution_clock::now();

                // Create an Arrow buffer from the vector
                auto arrowBuffer = arrow::Buffer::Wrap(data.data(), data.size());
                // Create an Arrow BufferReader from the Arrow buffer
                auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);


                // Open Parquet file reader
                std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);
                if (!status.ok()) {
                    std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
                    return;
                }

                auto end_buffer = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> buffer_duration = end_buffer - start_buffer;
                std::cout << "Time to preparse arrow buffer and arrowreader: " << buffer_duration.count() << " seconds" << std::endl;


////////////////////////////////////////////////
                
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


        /////////////////////////////////////////////////
                
                auto start_stats = std::chrono::high_resolution_clock::now();

                // Ensure there is at least one row group
                if (arrowReader->num_row_groups() == 0) {
                    std::cerr << "No row groups found in the Parquet file." << std::endl;
                    return;
                }

                // Find schema and find ID of parsed column
                std::shared_ptr<arrow::Schema> schema;
                arrowReader->GetSchema(&schema);
                int column_index = -1;
                for (int i = 0; i < schema->num_fields(); ++i) {
                    if (schema->field(i)->name() == columnName) {
                        column_index = i;
                        break;
                    }
                }

                if (column_index == -1) {
                    std::cerr << "Column  not found in the schema." << std::endl;
                    return;
                }

                std::vector<int> matching_row_groups;

                for (int row_group_index = 0; row_group_index < arrowReader->num_row_groups(); ++row_group_index) {
                    auto metadata = arrowReader->parquet_reader()->metadata();
                    auto row_group_metadata = metadata->RowGroup(row_group_index);
                    auto column_metadata = row_group_metadata->ColumnChunk(column_index);
    
                    if (column_metadata->is_stats_set()) {
                        auto stats = column_metadata->statistics();
                        if (stats->HasMinMax()) {
                        // Assuming the ID column is of integer type; adjust the type as necessary
                        // Need to check the type of filter columns
                        int64_t min_value = static_cast<const parquet::Int64Statistics*>(stats.get())->min();
                        int64_t max_value = static_cast<const parquet::Int64Statistics*>(stats.get())->max();

                            if (min_value <= l_value && max_value >= l_value) {
                                matching_row_groups.push_back(row_group_index);
                                //std::cout << "Matching Row Group: " << row_group_index << std::endl;  // Print matching row group index
                            }
                        }
                    }
                }

                auto end_stats = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> stats_duration = end_stats - start_stats;
                std::cout << "Time to read statistics and find row groups: " << stats_duration.count() << " seconds" << std::endl; 
///////////////////////////////////////////////////////////

                auto start_table = std::chrono::high_resolution_clock::now();

                std::shared_ptr<arrow::Table> concatenated_table;

                if (!matching_row_groups.empty()) {


                    std::vector<std::shared_ptr<arrow::Table>> tables;
                    for (int row_group_index : matching_row_groups) {
                        std::shared_ptr<arrow::Table> table;
                        auto status = arrowReader->ReadRowGroup(row_group_index, &table);
                        if (!status.ok()) {
                            std::cerr << "Error reading Arrow table from RowGroup " << row_group_index << ": " << status.ToString() << std::endl;
                            continue;
                        }
                        tables.push_back(table);
                    }

                    // Assuming you want to concatenate all matching tables into a single table
                    // Assuming 'tables' is a std::vector<std::shared_ptr<arrow::Table>> containing your tables
                    arrow::Result<std::shared_ptr<arrow::Table>> concatenated_table_result = arrow::ConcatenateTables(tables);

                    if (!concatenated_table_result.ok()) {
                        // Handle error
                        std::cerr << "Failed to concatenate tables: " << concatenated_table_result.status() << std::endl;
                        return;
                    }

                    concatenated_table = *concatenated_table_result;


                }else{

                    auto t_status = arrowReader->ReadTable(&concatenated_table);
                    if (!t_status.ok()) {
                        std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
                    return;
                    }


                }

                auto end_table = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> table_duration = end_table - start_table;
                std::cout << "Time to read table: " << buffer_duration.count() << " seconds" << std::endl;  
////////////////////////////////////////////////////

                auto start_filter = std::chrono::high_resolution_clock::now();
                auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(concatenated_table);

                // 2: Build ScannerOptions for a Scanner to do a basic filter operation
                auto options = std::make_shared<arrow::dataset::ScanOptions>();


                // Build the Scanner
                auto builder = arrow::dataset::ScannerBuilder(dataset);
                // Set the filter
                arrow::Status build_status = builder.Filter(filter_expression);
                if (!build_status.ok()) {
                    std::cerr << "Failed to apply filter: " << status.ToString() << std::endl;
                    return;
                }

                auto scanner = builder.Finish();

                // Perform the Scan and retrieve filtered result as Table
                auto result_table = scanner.ValueOrDie()->ToTable();

                auto end_filter = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> filter_duration = end_filter - start_filter;
                std::cout << "Time to wrap in dataset and filter: " << filter_duration.count() << " seconds" << std::endl; 
//

/////////////////////////////////////////////////////

                // Print the Arrow table's schema
                //std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;
                // Print the Arrow table's data
                //std::cout << "Table Data:\n" << result_table.ValueUnsafe()->ToString() << std::endl;

                // Assuming 'filtered_result_json' contains the filtered data in JSON format
                std::string filtered_result_json = result_table.ValueUnsafe()->ToString();


                // Send the JSON response
                response.send(Http::Code::Ok, filtered_result_json, MIME(Application, Json));
                // Send the JSON response
                //response.send(Http::Code::Ok, "Done");
            } else {
                // Error handling
                std::cerr << "Failed to get object: " << getObjectOutcome.GetError().GetMessage() << std::endl;
                response.send(Http::Code::Internal_Server_Error, "Failed to get object from S3");
            }
        } catch (const std::exception& e) {
            // Exception handling
            std::cerr << "Exception: " << e.what() << std::endl;
            response.send(Http::Code::Internal_Server_Error, "Exception occurred");
        }

        // Shutdown AWS SDK
        Aws::ShutdownAPI(options);
    }
};

int main() {

    if (setenv("AWS_EC2_METADATA_DISABLED", "true", 1) != 0) {
        // Handle error if needed
    }
    Http::listenAndServe<S3Handler>(Pistache::Address("*:8080"));
    return 0;
}
