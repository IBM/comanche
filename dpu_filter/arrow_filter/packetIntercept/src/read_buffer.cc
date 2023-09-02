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


#include "read_buffer.h"


extern "C" {

void readBuffer(const unsigned char* data_buffer, size_t data_size) {
    
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
    }

    // Read entire file as a single Arrow table
    std::shared_ptr<arrow::Table> table;
    status = arrow_reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
    }

    // Display the Arrow table's schema
    std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    std::cout << "Table Data:\n" << table->ToString() << std::endl;

   // Write the Arrow table to a new Parquet file
    arrow::Result<std::shared_ptr<arrow::io::FileOutputStream>> out_stream_result =
        arrow::io::FileOutputStream::Open("newdata2.parquet");
    if (!out_stream_result.ok()) {
        std::cerr << "Error opening output Parquet file: " << out_stream_result.status().ToString() << std::endl;
        return;
    }

    std::shared_ptr<arrow::io::FileOutputStream> out_stream = out_stream_result.ValueOrDie();
    status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out_stream);
    if (!status.ok()) {
        std::cerr << "Error writing Arrow table to Parquet file: " << status.ToString() << std::endl;
        return;
    }

    out_stream->Close();
    std::cout << "Arrow table written to 'newdata.parquet'" << std::endl;

}

}

