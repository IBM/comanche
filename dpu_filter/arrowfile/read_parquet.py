#include <iostream>
#include <fstream>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/statistics.h>

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

    // Access the Parquet file's schema
    std::shared_ptr<arrow::Schema> schema = arrow_reader->parquet_reader()->metadata()->schema();

    // Iterate through schema fields and access statistics (if available)
    for (int i = 0; i < schema->num_fields(); ++i) {
        const auto& field = schema->field(i);
        const auto& column_stats = field->statistics();
        
        std::cout << "Field Name: " << field->name() << std::endl;
        
        // Check if statistics are available for the column
        if (column_stats) {
            std::cout << "Statistics:\n" << column_stats->ToString() << std::endl;
        } else {
            std::cout << "Statistics not available for this column." << std::endl;
        }
    }

    return 0;
}

