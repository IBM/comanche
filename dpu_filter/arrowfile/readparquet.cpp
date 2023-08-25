#include <iostream>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

int main() {
    // Path to the Parquet file
    const std::string path_to_file = "data.parquet";

    // Initialize Arrow
    arrow::Status status = arrow::Status::OK();

    // Open Parquet file for reading
    std::shared_ptr<arrow::io::RandomAccessFile> input;
    status = arrow::io::ReadableFile::Open(path_to_file).Value(&input);
    if (!status.ok()) {
        std::cerr << "Error opening Parquet file: " << status.ToString() << std::endl;
        return 1;
    }

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    status = parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &arrow_reader);
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

