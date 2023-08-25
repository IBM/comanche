#include <iostream>
#include <fstream>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

int main() {
    // Path to the Parquet file
    const std::string path_to_file = "data.parquet";

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

    // Display the Arrow table's schema
    std::cout << "Tablesss Schema:\n" << table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    std::cout << "Table Data:\n" << table->ToString() << std::endl;

    return 0;
}

/*#include <iostream>
#include <fstream>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

int main() {
    // Path to the Parquet file
    const std::string path_to_file = "data.parquet";

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

    // Create an Arrow RandomAccessFile from the Arrow buffer
    std::shared_ptr<arrow::io::RandomAccessFile> input =
        std::make_shared<arrow::io::BufferReader>(arrow_buffer);

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    status = parquet::arrow::OpenFile(input, pool, &arrow_reader);
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

    // Display the Arrow table's schema
    std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    std::cout << "Table Data:\n" << table->ToString() << std::endl;

    return 0;
}
*/