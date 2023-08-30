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

    std::cout << "Arrow table  " << std::endl;

    // Modify the Arrow table, e.g., filter rows
    // For simplicity, let's just use the first half of the table
    //auto sliced_table = table->Slice(0, table->num_rows() / 2);

    // Define the filtering condition
    // Create an InMemoryDataset from the original table
   // auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

    auto filter_date = std::chrono::system_clock::from_time_t(1678320000); // August 7, 2023
    auto timestamp_type = arrow::timestamp(arrow::TimeUnit::SECOND);
    auto filter_date_scalar = arrow::MakeScalar(timestamp_type, filter_date).ValueOrDie();

    std::cout << "Filter Date Scalar: " << filter_date_scalar->ToString() << std::endl;

    // 1: Wrap the Table in a Dataset so we can use a Scanner
    //just (table)?
    auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

    // 2: Build ScannerOptions for a Scanner to do a basic filter operation
    auto options = std::make_shared<arrow::dataset::ScanOptions>();
    options->filter = arrow::compute::less(
        arrow::compute::field_ref("timestamp"),
        arrow::compute::literal(filter_date_scalar));

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

    // Serialize the modified table to a Parquet buffer
    auto parquet_stream = arrow::io::BufferOutputStream::Create(1024).ValueOrDie();
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

