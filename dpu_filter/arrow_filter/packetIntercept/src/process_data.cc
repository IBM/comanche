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

#include "process_data.h" // Include your header file

extern "C" {


int processParquetFile(const char* filename) {
    // Your C++ code implementation here
    const std::string path_to_file = filename;

    std::cout << "Start" << std::endl;

    // Open the Parquet file for reading
    std::ifstream file_stream(path_to_file, std::ios::binary | std::ios::ate);
    if (!file_stream.is_open()) {
        std::cerr << "Error opening Parquet file" << std::endl;
        return 1;
    }

    // Get the size of the file
    std::streamsize file_size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);

    // Read the file content into a buffer
    std::vector<unsigned char> buffer(file_size);
    if (!file_stream.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        std::cerr << "Error reading Parquet file" << std::endl;
        return 1;
    }

    // Initialize Arrow
    arrow::Status status = arrow::Status::OK();

    // Create an Arrow memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    // Convert the buffer to an Arrow buffer
    auto arrow_buffer = std::make_shared<arrow::Buffer>(buffer.data(), file_size);

    // Create an Arrow BufferReader from the Arrow buffer
    auto buffer_reader = std::make_shared<arrow::io::BufferReader>(arrow_buffer);

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    status = parquet::arrow::OpenFile(buffer_reader, pool, &arrow_reader);
    if (!status.ok()) {
        std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
        return 1;
    }

    // Read entire file as a single Arrow table
    std::shared_ptr<arrow::Table> table;
    status = arrow_reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
        return 1;
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
        return 1;
    }
    auto filtered_table = result.ValueOrDie();

   

    // Display the Arrow table's schema
    std::cout << "Filtered Table Schema:\n" << filtered_table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    std::cout << "Filtered Table Data:\n" << filtered_table->ToString() << std::endl;


    return 0; // Return an appropriate value
}

}