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
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/statistics.h>
#include <arrow/table.h>
#include <arrow/result.h>
#include <arrow/compute/expression.h> // Include this header
#include "/home/ubuntu/json/include/nlohmann/json.hpp"
#include <gandiva/node.h>
#include <gandiva/tree_expr_builder.h>
#include <gandiva/projector.h>
#include <gandiva/filter.h>
#include <arrow/pretty_print.h>
#include <cstdlib>
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
        // Parse the SQL expression
        std::vector<Token> tokens = SQLParser::parse(sqlExpression);


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


                // Open Parquet file reader
                std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);
                if (!status.ok()) {
                    std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
                    return;
                }

                std::shared_ptr<arrow::Schema> schema;
                arrowReader->GetSchema(&schema);

///////////////////////////////////////////////
 

///////////////////////////////////////////


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
                }

                // Find the field in the schema
                auto field = schema->GetFieldByName(columnName); 

                // Dynamically create the field node based on the schema type
                std::shared_ptr<gandiva::Node> field_node;
                std::shared_ptr<gandiva::Node> literal_node;

                // Example of handling int32 and int64. Extend this as needed for other types.
                if (field->type()->id() == arrow::Type::INT32) {
                    int32_t l_value = std::stoi(value); // Convert value to int32_t
                    field_node = gandiva::TreeExprBuilder::MakeField(field);
                    literal_node = gandiva::TreeExprBuilder::MakeLiteral(l_value);
                } else if (field->type()->id() == arrow::Type::INT64) {
                    int64_t l_value = std::stoll(value); // Convert value to int64_t
                    field_node = gandiva::TreeExprBuilder::MakeField(field);
                    literal_node = gandiva::TreeExprBuilder::MakeLiteral(l_value);
                }
                // Add more else if blocks here for other types like FLOAT, STRING, etc.

                std::shared_ptr<gandiva::Condition> condition;

                // Build Gandiva condition based on the operator
                if (operatorSymbol == "=") {
                    condition = gandiva::TreeExprBuilder::MakeCondition(gandiva::TreeExprBuilder::MakeFunction("equal", {field_node, literal_node}, arrow::boolean()));
                } else if (operatorSymbol == ">") {
                    condition = gandiva::TreeExprBuilder::MakeCondition(gandiva::TreeExprBuilder::MakeFunction("greater_than", {field_node, literal_node}, arrow::boolean()));
                } else if (operatorSymbol == ">=") {
                    condition = gandiva::TreeExprBuilder::MakeCondition(gandiva::TreeExprBuilder::MakeFunction("greater_than_or_equal_to", {field_node, literal_node}, arrow::boolean()));
                } else if (operatorSymbol == "<") {
                    condition = gandiva::TreeExprBuilder::MakeCondition(gandiva::TreeExprBuilder::MakeFunction("less_than", {field_node, literal_node}, arrow::boolean()));
                } else if (operatorSymbol == "<=") {
                    condition = gandiva::TreeExprBuilder::MakeCondition(gandiva::TreeExprBuilder::MakeFunction("less_than_or_equal_to", {field_node, literal_node}, arrow::boolean()));
                } else if (operatorSymbol == "!=") {
                    condition = gandiva::TreeExprBuilder::MakeCondition(gandiva::TreeExprBuilder::MakeFunction("not_equal", {field_node, literal_node}, arrow::boolean()));
                }

/////////////////////////////////////////////////

                std::shared_ptr<gandiva::Filter> gandiva_filter;
                auto filter_status = gandiva::Filter::Make(schema, condition, &gandiva_filter);
                if (!filter_status.ok()) {
                    std::cerr << "Error creating Gandiva filter: " << filter_status.message() << std::endl;
                    response.send(Http::Code::Internal_Server_Error, "Error applying filter");
                    return;
                }

////////////////////////////////////////////////
/*Stats based row group filtering*/

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


                                        // Example of handling int32 and int64. Extend this as needed for other types.
                            if (field->type()->id() == arrow::Type::INT32) {
                                int32_t l_value = std::stoi(value); // Convert value to int32_t
                                int32_t min_value = static_cast<const parquet::Int32Statistics*>(stats.get())->min();
                                int32_t max_value = static_cast<const parquet::Int32Statistics*>(stats.get())->max();

                                if (min_value <= l_value && max_value >= l_value) {
                                    matching_row_groups.push_back(row_group_index);
                                //std::cout << "Matching Row Group: " << row_group_index << std::endl;  // Print matching row group index
                                }  
                            } else if (field->type()->id() == arrow::Type::INT64) {
                                int64_t l_value = std::stoll(value); // Convert value to int64_t
                                int64_t min_value = static_cast<const parquet::Int64Statistics*>(stats.get())->min();
                                int64_t max_value = static_cast<const parquet::Int64Statistics*>(stats.get())->max();

                                if (min_value <= l_value && max_value >= l_value) {
                                    matching_row_groups.push_back(row_group_index);
                                //std::cout << "Matching Row Group: " << row_group_index << std::endl;  // Print matching row group index
                                }
                            }

                        }
                    }
                }



///////////////////////////////////////////////
//Read Table

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

///////////////////////////////////////////////

                arrow::TableBatchReader reader(*concatenated_table);
                auto pool = arrow::default_memory_pool();

                std::shared_ptr<arrow::RecordBatch> batch;

                std::vector<std::shared_ptr<arrow::RecordBatch>> filtered_batches;


                while (reader.ReadNext(&batch).ok() && batch != nullptr) {
                    // Prepare the memory pool and selection vector

                    std::shared_ptr<gandiva::SelectionVector> result_indices;
                    auto status = gandiva::SelectionVector::MakeInt32(static_cast<int>(batch->num_rows()), pool, &result_indices);
                    if (!status.ok()) {
                        std::cerr << "Error creating selection vector: " << status.message() << std::endl;
                        continue;  // Or handle error appropriately
                    }

                    // Evaluate the Gandiva filter on the batch
                    status = gandiva_filter->Evaluate(*batch, result_indices);
                    if (!status.ok()) {
                        std::cerr << "Filter application failed: " << status.message() << std::endl;
                        continue;  // Or handle error appropriately
                    }

                    // Use the results: Convert the selection vector to an Array for 'Take'
                    auto take_indices = result_indices->ToArray();
                    arrow::Datum maybe_filtered_batch;
                    auto maybe_result = arrow::compute::Take(arrow::Datum(batch), arrow::Datum(take_indices), arrow::compute::TakeOptions::Defaults());

                    if (!maybe_result.ok()) {
                        // Handle the failure, e.g., log it, and decide on the fly how to react.
                        std::cerr << "Failed to apply Take operation: " << maybe_result.status().message() << std::endl;
                        // Choose what to do: return, continue with next item in loop, etc.
                    } else {
                        maybe_filtered_batch = std::move(maybe_result).ValueOrDie();
                        auto filtered_batch = maybe_filtered_batch.record_batch();

                        // Accumulate the filtered batches
                        if (filtered_batch != nullptr) {
                            //COncatenate
                            filtered_batches.push_back(filtered_batch);
                            /*std::stringstream ss;
                            arrow::PrettyPrint(*filtered_batch, {}, &ss);
                            std::cout << ss.str() << std::endl;*/
                        }
                    }
                }


                // After the loop, concatenate all filtered RecordBatches into a single Table
            
                arrow::Result<std::shared_ptr<arrow::Table>> result = arrow::Table::FromRecordBatches(filtered_batches);
                std::shared_ptr<arrow::Table> result_table = result.ValueOrDie();

                // Print the Arrow table's schema
                //std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;
                // Print the Arrow table's data
               // std::cout << "Table Data:\n" << result_table.ValueUnsafe()->ToString() << std::endl;

                // Assuming 'filtered_result_json' contains the filtered data in JSON format
                std::string filtered_result_json = result_table->ToString();

                // Send the JSON response
                response.send(Http::Code::Ok, filtered_result_json, MIME(Application, Json));
            
                //response.send(Http::Code::Ok, "jo");
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
