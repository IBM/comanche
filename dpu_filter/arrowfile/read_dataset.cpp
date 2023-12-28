#include <iostream>
#include <vector>
#include <fstream>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <chrono>
#include <cstdlib> // Include this header for string conversion

std::string processParquetFile(const char* filename, int filter_id) {
    const std::string path_to_file = filename;

    std::cout << "Start" << std::endl;
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

    // Extract the "ID" column from the table
    auto id_column = table->GetColumnByName("ID");
    if (!id_column) {
        std::cerr << "Error extracting ID column: Column not found." << std::endl;
        return "";
    }

    // Create a new table with only the "ID" column
    std::shared_ptr<arrow::Table> filtered_table = arrow::Table::Make(table->schema(), {id_column});

    // Display the Arrow table's data (ID column)
    std::cout << "Filtered Column Data:\n" << filtered_table->ToString() << std::endl;
    std::cout << "Filtered Table Schema:\n" << filtered_table->schema()->ToString() << std::endl;


    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string output_filename = "filtered_output_" + std::to_string(filter_id) + ".parquet";

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
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <parquet_filename> <Filter_id>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];
    int filter_id = std::atoi(argv[2]);

    std::string output_filename = processParquetFile(filename, filter_id);

    if (!output_filename.empty()) {
        std::cout << "Processing successful. Output file: " << output_filename << std::endl;
    } else {
        std::cerr << "Processing failed." << std::endl;
    }

    return output_filename.empty() ? 1 : 0;
}
