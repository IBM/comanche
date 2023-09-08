#include <iostream>
#include <vector>
#include <fstream> 
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

extern "C" {

unsigned char* processParquetFile(const char* filename, size_t* buffer_size) {
    // Your C++ code implementation here
    const std::string path_to_file = filename;

    std::cout << "Start" << std::endl;

    // Open the Parquet file for reading
    std::ifstream file_stream(path_to_file, std::ios::binary | std::ios::ate);
    if (!file_stream.is_open()) {
        std::cerr << "Error opening Parquet file" << std::endl;
        return nullptr;
    }

    // Get the size of the file
    std::streamsize file_size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);

    // Allocate a buffer to hold the file content
    unsigned char* buffer = new unsigned char[file_size];

    // Read the file content into the buffer
    if (!file_stream.read(reinterpret_cast<char*>(buffer), file_size)) {
        std::cerr << "Error reading Parquet file" << std::endl;
        delete[] buffer;
        return nullptr;
    }

    // Set the buffer size
    *buffer_size = static_cast<size_t>(file_size);

    return buffer;
}

}

/*int main() {
    const char* filename = "example.parquet";
    size_t buffer_size = 0;
    
    unsigned char* buffer = processParquetFile(filename, &buffer_size);
    if (buffer) {
        std::cout << "Successfully read Parquet file. Buffer size: " << buffer_size << std::endl;

        // You can now work with the buffer containing the Parquet file data

        delete[] buffer; // Don't forget to free the allocated memory
    } else {
        std::cerr << "Error processing Parquet file." << std::endl;
    }

    return 0;
}*/
