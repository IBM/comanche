#include <iostream>
#include <vector>
#include <fstream> 
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
#include <arrow/builder.h>


std::string processParquetFile(const char* filename) {
    // Your C++ code implementation here
    const std::string path_to_file = filename;

    std::cout << "Start" << std::endl;
    // Open Parquet file reader
   // Open Parquet file reader
    arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> readable_file =
        arrow::io::ReadableFile::Open(path_to_file);
    if (!readable_file.ok()) {
        std::cerr << "Error opening Parquet file: " << readable_file.status().ToString() << std::endl;
        return "";
    }

    arrow::Status status = arrow::Status::OK();

    // Create an Arrow memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    status = parquet::arrow::OpenFile(readable_file.ValueOrDie(), pool, &arrow_reader);
    if (!status.ok()) {
        std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
        return "";
    }
    

    // Read entire file as a single Arrow table
    std::shared_ptr<arrow::Table> table;
    status = arrow_reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
        return "";
    }

    // Define the filtering condition
    auto filter_id = 120;
    auto id_field = arrow::compute::field_ref("ID");
    auto filter_id_scalar = arrow::compute::literal(filter_id);


    auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);


    // 2: Build ScannerOptions for a Scanner to do a basic filter operation
    auto options = std::make_shared<arrow::dataset::ScanOptions>();
    options->filter = arrow::compute::less(id_field, filter_id_scalar);



    // 3: Build the Scanner
    auto builder = arrow::dataset::ScannerBuilder(dataset, options);
    auto scanner = builder.Finish();

     

    // 4: Perform the Scan and make a Table with the result
    arrow::Result<std::shared_ptr<arrow::Table>> result = scanner.ValueUnsafe()->ToTable();
    if (!result.ok()) {
        std::cerr << "Error filtering Arrow Table: " << result.status().ToString() << std::endl;
        return "";
    }
    auto filtered_table = result.ValueOrDie();

   

    // Display the Arrow table's schema
    //std::cout << "Filtered Table Schema:\n" << filtered_table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    //std::cout << "Filtered Table Data:\n" << filtered_table->ToString() << std::endl;
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string output_filename = "filtered_output_" + std::to_string(now_time) + ".parquet";

    // Serialize the filtered table to the output Parquet file
    status = parquet::arrow::WriteTable(*filtered_table, arrow::default_memory_pool(),
                                        arrow::io::FileOutputStream::Open(output_filename).ValueOrDie());
    if (!status.ok()) {
        std::cerr << "Error writing filtered Arrow table to Parquet file: " << status.ToString() << std::endl;
        return "";
    }

    return output_filename;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <parquet_filename>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];
    
    
    std::string output_filename = processParquetFile(filename);

    if (!output_filename.empty()) {
        std::cout << "Processing successful. Output file: " << output_filename << std::endl;
    } else {
        std::cerr << "Processing failed." << std::endl;
    }

    return output_filename.empty() ? 1 : 0;
}