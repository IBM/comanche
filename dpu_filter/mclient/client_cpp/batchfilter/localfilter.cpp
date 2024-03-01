#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/pretty_print.h>
#include <parquet/arrow/reader.h>
#include <iostream>
#include <chrono>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h> // Include the direct Parquet file reader
#include <gandiva/node.h>
#include <gandiva/tree_expr_builder.h>
#include <gandiva/projector.h>
#include <gandiva/filter.h>

int main() {
    // Initialize S3
    auto start_time = std::chrono::high_resolution_clock::now();
    auto status = arrow::fs::InitializeS3(arrow::fs::S3GlobalOptions());
    if (!status.ok()) {
        std::cerr << "Failed to initialize S3 subsystem: " << status.ToString() << std::endl;
        return -1;
    }

    // Set up S3 options and create an S3FileSystem
    arrow::fs::S3Options options = arrow::fs::S3Options::FromAccessKey("minioadmin", "minioadmin");
    options.endpoint_override = "10.10.10.18:9000";
    options.scheme = "https";
    options.background_writes = true;

    auto s3fs_result = arrow::fs::S3FileSystem::Make(options);
    if (!s3fs_result.ok()) {
        std::cerr << "Could not create S3FileSystem: " << s3fs_result.status().ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }
    auto s3fs = *s3fs_result;

    // Specify the file path in S3
    std::string file_path = "mycsvbucket/sampledata/dataStat_1000000.parquet";

    // Open the file for reading
    auto file_result = s3fs->OpenInputFile(file_path);
    if (!file_result.ok()) {
        std::cerr << "Could not open file: " << file_result.status().ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }
    auto file = *file_result;

    // Create a Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
    status = parquet::arrow::OpenFile(file, arrow::default_memory_pool(), &parquet_reader);
    if (!status.ok()) {
        std::cerr << "Could not open Parquet file: " << status.ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }

    auto num_row_groups = parquet_reader->num_row_groups();
    std::shared_ptr<arrow::Schema> schema;
    parquet_reader->GetSchema(&schema);

    std::vector<int> row_groups(num_row_groups);
    std::iota(row_groups.begin(), row_groups.end(), 0); // Fill with all row group indices

    //std::vector<int> row_groups(1); // Create a vector to hold one row group index
    //row_groups[0] = 0; // Set the index of the first row group

    // Create RecordBatchReader for specified row groups
    
    // Initialize RecordBatchReader outside the conflicting scope
   
    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    auto stat = parquet_reader->GetRecordBatchReader(row_groups, &batch_reader);
    if (!stat.ok()) {
        std::cerr << "Could not create RecordBatchReader: " << stat.ToString() << std::endl;
        return -1;
    }


///////////////////////////
/*Filtering*/
    std::string columnName = "ID";
    std::string operatorSymbol = "<";
    std::string value = "120";


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

        
    std::shared_ptr<gandiva::Filter> gandiva_filter;
    auto filter_status = gandiva::Filter::Make(schema, condition, &gandiva_filter);
    if (!filter_status.ok()) {
        std::cerr << "Error creating Gandiva filter: " << filter_status.message() << std::endl;
    }
//////////////////////////

    // Stream through record batches
    std::shared_ptr<arrow::RecordBatch> batch;
    auto pool = arrow::default_memory_pool();
    std::vector<std::shared_ptr<arrow::RecordBatch>> filtered_batches;
    
    auto start_table = std::chrono::high_resolution_clock::now();

    while (batch_reader->ReadNext(&batch).ok() && batch != nullptr) {
        // Process each batch here
        //std::cout << "Processed a batch with " << batch->num_rows() << " rows." << std::endl;

        std::shared_ptr<gandiva::SelectionVector> result_indices;
        auto status = gandiva::SelectionVector::MakeInt32(static_cast<int>(batch->num_rows()), pool, &result_indices);
        if (!status.ok()) {
            std::cerr << "Error creating selection vector: " << status.message() << std::endl;
            continue;
        }

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


                    auto end_table = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> table_duration = end_table - start_table;
                std::cout << "Time to read table: " << table_duration.count() << " seconds" << std::endl;  

                 arrow::Result<std::shared_ptr<arrow::Table>> result = arrow::Table::FromRecordBatches(filtered_batches);
                std::shared_ptr<arrow::Table> result_table = result.ValueOrDie();

            //std::cout << "Table Data:\n" << result_table->ToString() << std::endl;


    // Finalize S3 subsystem when done
    status = arrow::fs::FinalizeS3();
    if (!status.ok()) {
        std::cerr << "Error finalizing S3 subsystem: " << status.ToString() << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;
    std::cout << "Total execution time: " << total_duration.count() << " seconds" << std::endl;

    return 0;
}
