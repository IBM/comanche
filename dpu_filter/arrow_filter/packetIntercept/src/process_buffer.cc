#include <iostream>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "process_buffer.h"


extern "C" {

unsigned char* processParquetData(const unsigned char* data_buffer, size_t data_size, size_t* filtered_buffer_size) {
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

    // Modify the Arrow table, e.g., filter rows
    // For simplicity, let's just use the first half of the table
    auto sliced_table = table->Slice(0, table->num_rows() / 2);

    // Serialize the modified table to a Parquet buffer
    auto parquet_stream = arrow::io::BufferOutputStream::Create(1024).ValueOrDie();
    status = parquet::arrow::WriteTable(*sliced_table, arrow::default_memory_pool(), parquet_stream);
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

