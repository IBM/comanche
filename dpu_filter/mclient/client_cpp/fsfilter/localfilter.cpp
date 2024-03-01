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
#include <arrow/compute/expression.h> // Include this header

int main() {
    // Initialize S3
    auto start_time = std::chrono::high_resolution_clock::now();
    auto status = arrow::fs::InitializeS3(arrow::fs::S3GlobalOptions());
    if (!status.ok()) {
        std::cerr << "Failed to initialize S3 subsystem: " << status.ToString() << std::endl;
        return -1;
    }

    auto start_s3_fetch = std::chrono::high_resolution_clock::now(); 
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

    auto end_s3_fetch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> s3_fetch_duration = end_s3_fetch - start_s3_fetch;
    std::cout << "Time to make filesystem: " << s3_fetch_duration.count() << " seconds" << std::endl;

    // Specify the file path in S3
    std::string file_path = "mycsvbucket/sampledata/dataStat_1000000.parquet";

    // Open the file for reading using the new Result pattern
    auto file_result = s3fs->OpenInputFile(file_path);
    if (!file_result.ok()) {
        std::cerr << "Could not open file: " << file_result.status().ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }
    auto file = *file_result;

    // Read the Parquet file into an Arrow Table
    std::unique_ptr<parquet::arrow::FileReader> reader;
    status = parquet::arrow::OpenFile(file, arrow::default_memory_pool(), &reader);
    if (!status.ok()) {
        std::cerr << "Could not open Parquet file: " << status.ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }

    auto start_table = std::chrono::high_resolution_clock::now();
    /*std::shared_ptr<arrow::Table> table;
    status = reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Could not read table from Parquet file: " << status.ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }*/

        // Read only the first row group instead of the entire table
    std::shared_ptr<arrow::Table> table;
    if (reader->num_row_groups() > 0) {
        status = reader->ReadRowGroup(0, &table); // Reading only the first row group
        if (!status.ok()) {
            std::cerr << "Could not read the first row group: " << status.ToString() << std::endl;
            arrow::fs::FinalizeS3();
            return -1;
        }
    } else {
        std::cerr << "The Parquet file has no row groups." << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }
    

    auto end_table = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> table_duration = end_table - start_table;
    std::cout << "Time to read table: " << table_duration.count() << " seconds" << std::endl;  

    std::string columnName = "ID";
    auto field_ref = arrow::compute::field_ref(columnName);

    int l_value = 120;
    auto literal_value = arrow::compute::literal(l_value);

    arrow::compute::Expression filter_expression;
    filter_expression = arrow::compute::less_equal(field_ref,literal_value);

    std::shared_ptr<arrow::dataset::Dataset> dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);


                // Build the Scanner
    auto builder = arrow::dataset::ScannerBuilder(dataset);     
                // Set the filter
    arrow::Status build_status = builder.Filter(filter_expression);

    auto scanner = builder.Finish();

    // Perform the Scan and retrieve filtered result as Table
    auto result_table = scanner.ValueOrDie()->ToTable();

    std::cout << "Table Data:\n" << result_table.ValueUnsafe()->ToString() << std::endl;


    // Finalize S3 subsystem when done
    status = arrow::fs::FinalizeS3();
    if (!status.ok()) {
        std::cerr << "Error finalizing S3 subsystem: " << status.ToString() << std::endl;
    }


    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;
    std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;  

    return 0;
}
