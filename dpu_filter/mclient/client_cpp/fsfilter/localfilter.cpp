#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/io/api.h>
#include <arrow/pretty_print.h>
#include <parquet/arrow/reader.h>
#include <iostream>
#include <chrono>
#include <arrow/pretty_print.h>

int main() {
    // Initialize S3

    auto start_time = std::chrono::high_resolution_clock::now();
    arrow::Status status = arrow::fs::InitializeS3(arrow::fs::S3GlobalOptions());
    if (!status.ok()) {
        std::cerr << "Failed to initialize S3 subsystem: " << status.ToString() << std::endl;
        return -1;
    }

    

    // Set up S3 options and create an S3FileSystem
    arrow::fs::S3Options options = arrow::fs::S3Options::FromAccessKey("minioadmin", "minioadmin");
    options.endpoint_override = "10.10.10.18:9000";
    options.scheme = "https";


    auto start_s3_fetch = std::chrono::high_resolution_clock::now(); 

    //Bellow is the function that takes the most time
    auto s3fs_result = arrow::fs::S3FileSystem::Make(options);
    if (!s3fs_result.ok()) {
        std::cerr << "Could not create S3FileSystem: " << s3fs_result.status().ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }

    
    std::shared_ptr<arrow::fs::FileSystem> s3fs = *s3fs_result;

    // Setup FileSelector
    arrow::fs::FileSelector selector;
    selector.base_dir = "mycsvbucket/sampledata";
    selector.recursive = false;

    
    
    

    // Create a dataset factory
    auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();
    arrow::dataset::FileSystemFactoryOptions fs_factory_options;
    
    

    auto dataset_factory_result = arrow::dataset::FileSystemDatasetFactory::Make(s3fs, selector, format, fs_factory_options);
    if (!dataset_factory_result.ok()) {
        std::cerr << "Could not create dataset factory: " << dataset_factory_result.status().ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }
    auto end_s3_fetch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> s3_fetch_duration = end_s3_fetch - start_s3_fetch;
    std::cout << "Time to fetch object from S3: " << s3_fetch_duration.count() << " seconds" << std::endl;

    auto dataset_factory = *dataset_factory_result;
    auto dataset_result = dataset_factory->Finish();
    if (!dataset_result.ok()) {
        std::cerr << "Could not finalize dataset: " << dataset_result.status().ToString() << std::endl;
        arrow::fs::FinalizeS3();
        return -1;
    }

    auto dataset = *dataset_result;



    std::string columnName = "ID";
    auto field_ref = arrow::compute::field_ref(columnName);

    int l_value = 120;
    auto literal_value = arrow::compute::literal(l_value);

    arrow::compute::Expression filter_expression;
    filter_expression = arrow::compute::less_equal(field_ref,literal_value);
           // Build ScannerOptions for a Scanner to apply filter operation
    auto ptions = std::make_shared<arrow::dataset::ScanOptions>();

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
