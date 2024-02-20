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
                
                // Read the stream into a vector: takes long time
                //std::vector<char> data(std::istreambuf_iterator<char>(objectStream), {});

/////////////////////// Faster way Estimate the size of the stream to reserve vector capacity upfront (optional)
                std::vector<char> data;
                
                objectStream.seekg(0, std::ios::end);
                std::streamsize size = objectStream.tellg();
                objectStream.seekg(0, std::ios::beg);

                if (size > 0) {
                    data.reserve(static_cast<size_t>(size));
                }

                // Use a buffer to read chunks of the stream
                constexpr std::streamsize bufferSize = 8192*8; // Example buffer size, adjust as needed
                char buffer[bufferSize];

                while (objectStream.read(buffer, bufferSize) || objectStream.gcount() > 0) {
                    data.insert(data.end(), buffer, buffer + objectStream.gcount());
                }

                // Create an Arrow buffer from the vector
                auto arrowBuffer = arrow::Buffer::Wrap(data.data(), data.size());
                // Create an Arrow BufferReader from the Arrow buffer
                auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);

///////////////////////////////////////////////
                // Open Parquet file reader
                std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);
                if (!status.ok()) {
                    std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
                    return;
                }

                // Ensure there is at least one row group
                if (arrowReader->num_row_groups() == 0) {
                    std::cerr << "No row groups found in the Parquet file." << std::endl;
                    return;
                }
                // Define the filtering condition, hardcoded, need to parse from SQL statement
                int filter_id = 120;
                auto id_field = arrow::compute::field_ref("ID");
                auto filter_id_scalar = arrow::compute::literal(filter_id);

                //Need to read schema to determine column ID, then based on schema, find row groups



                // Find schema and find ID of parsed column
                std::shared_ptr<arrow::Schema> schema;
                arrowReader->GetSchema(&schema);
                int column_index = -1;
                for (int i = 0; i < schema->num_fields(); ++i) {
                    if (schema->field(i)->name() == "ID") {
                        column_index = i;
                        break;
                    }
                }

                if (column_index == -1) {
                    std::cerr << "Column 'ID' not found in the schema." << std::endl;
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

                            if (min_value <= filter_id && max_value >= filter_id) {
                                matching_row_groups.push_back(row_group_index);
                                //std::cout << "Matching Row Group: " << row_group_index << std::endl;  // Print matching row group index
                            }
                        }
                    }
                }

                if (matching_row_groups.empty()) {
                    std::cerr << "No matching row groups found." << std::endl;
                    return;
                }

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

                std::shared_ptr<arrow::Table> concatenated_table = *concatenated_table_result;


                auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(concatenated_table);

                // 2: Build ScannerOptions for a Scanner to do a basic filter operation
                auto options = std::make_shared<arrow::dataset::ScanOptions>();
                options->filter = arrow::compute::less(id_field, filter_id_scalar);

                // 3: Build the Scanner
                auto builder = arrow::dataset::ScannerBuilder(dataset, options);
                auto scanner = builder.Finish();

                // Perform the Scan and retrieve filtered result as Table
                auto result_table = scanner.ValueOrDie()->ToTable();

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
    Http::listenAndServe<S3Handler>(Pistache::Address("*:8080"));
    return 0;
}
