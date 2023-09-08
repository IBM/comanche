#include <iostream>
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
#include <arrow/compute/expression.h> // Include this header


#include "process_buffer.h"


extern "C" {

unsigned char* processParquetData(const unsigned char* data_buffer, size_t data_size, size_t* filtered_buffer_size) {
    
    std::cout << "Arrow table  " << std::endl;
    // Initialize Arrow
    arrow::Status status = arrow::Status::OK();

    // Create an Arrow memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    // Convert the data buffer to an Arrow buffer
    auto arrow_buffer = std::make_shared<arrow::Buffer>(data_buffer, data_size);

    // Create an Arrow BufferReader from the Arrow buffer
    auto buffer_reader = std::make_shared<arrow::io::BufferReader>(arrow_buffer);

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    status = parquet::arrow::OpenFile(buffer_reader, pool, &arrow_reader);
    if (!status.ok()) {
        std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
        return nullptr;
    }

    // Read entire file as a single Arrow table
    std::shared_ptr<arrow::Table> table;
    status = arrow_reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
        return nullptr;
    }

    // Display the Arrow table's schema
    //std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    //std::cout << "Table Data:\n" << table->ToString() << std::endl;


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
        return nullptr;
    }
    auto filtered_table = result.ValueOrDie();

    // Display the Arrow table's schema
    //std::cout << "Filtered Table Schema:\n" << filtered_table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    //std::cout << "Table Data:\n" << filtered_table->ToString() << std::endl;
    std::cout << "Table Data:\n" << std::endl;

 // Serialize the modified table to a Parquet buffer
auto parquet_stream = arrow::io::BufferOutputStream::Create().ValueOrDie();
status = parquet::arrow::WriteTable(*filtered_table, arrow::default_memory_pool(), parquet_stream);
if (!status.ok()) {
    std::cerr << "Error writing modified Arrow table to Parquet buffer: " << status.ToString() << std::endl;
    return nullptr;
}
parquet_stream->Close();

// Get the Parquet buffer from the stream
arrow::Result<std::shared_ptr<arrow::Buffer>> parquet_result = parquet_stream->Finish();
if (!parquet_result.ok()) {
    std::cerr << "Error finishing Parquet stream: " << parquet_result.status().ToString() << std::endl;
    return nullptr;
}
std::shared_ptr<arrow::Buffer> parquet_buffer = parquet_result.ValueOrDie();
*filtered_buffer_size = parquet_buffer->size();

unsigned char* filtered_buffer = static_cast<unsigned char*>(malloc(*filtered_buffer_size));
memcpy(filtered_buffer, parquet_buffer->data(), *filtered_buffer_size);

return filtered_buffer;
}

}

