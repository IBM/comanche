#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <curl/curl.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/stream_writer.h>
#include <vector>
#include <nlohmann/json.hpp>  // Include the JSON library for processing JSON responses
#include <iomanip>  // for std::setprecision

using json = nlohmann::json;

// Helper function to encapsulate the static offset
static size_t& GetCurrentOffset() {
    static size_t currentOffset = 0;  // This maintains the offset
    return currentOffset;
}

void ResetWriteMemoryCallbackOffset() {
    size_t& currentOffset = GetCurrentOffset();  // Get reference to the static offset
    currentOffset = 0;  // Reset to zero
}

// Callback function for writing data received from the server into memory
size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto& memory = *static_cast<std::vector<char>*>(userp);
    size_t totalSize = size * nmemb;
    size_t& currentOffset = GetCurrentOffset(); // Maintains the current offset where data is to be written

    // Ensure there's enough space in the buffer
    /*if (currentOffset + totalSize > memory.size()) {
        std::cerr << "Buffer overflow detected! Resizing buffer." << std::endl;
        memory.resize(currentOffset + totalSize);
    }*/

    // Copy the received data into the vector at the current offset
    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, memory.begin() + currentOffset);
    currentOffset += totalSize; // Update the offset

    return totalSize;
}



// Function to trim surrounding quotes from a string
std::string TrimQuotes(const std::string& str) {
    if (str.length() >= 2 && str.front() == '"' && str.back() == '"') {
        return str.substr(1, str.length() - 2);
    }
    return str;
}

// Function to read SQL queries from a file and trim surrounding quotes
std::vector<std::string> readQueriesFromFile(const std::string& filename) {
    std::vector<std::string> queries;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Trim quotes when reading the query
            queries.push_back(TrimQuotes(line));
        }
    }

    return queries;
}

int main() {
    // URL and payload data
    std::string url = "http://10.10.10.20:8080/data";

    // Headers
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Initialize CURL
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return 1;
    }

    // Read SQL queries from an external file
    std::vector<std::string> queries = readQueriesFromFile("../query10.txt");

    std::ofstream timingFile;
    bool isFirstQuery = true;
    int queryIndex = 1;

    for (const auto& sqlQuery : queries) {
        std::cout << "Processing query #" << queryIndex << ": " << sqlQuery << std::endl;

        // Construct the payload with the current SQL query
        std::string payload = R"({
            "bucket": "encrypted",
            "key": "enc_web_sales.parquet",
            "output": "parquet",
            "sql": ")" + sqlQuery + R"("
        })";

        std::cout << "Payload: " << payload << std::endl;

        // Variables to store the response and timing
        std::vector<char> response_memory(1024 * 1024 * 1024); // Initialize with a large size to accommodate expected data
        ResetWriteMemoryCallbackOffset(); // Reset offset before starting

        auto start_time = std::chrono::high_resolution_clock::now();

        // Set CURL options
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_memory);
        curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 2 * 1024 * 1024); // Increased buffer size
        curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // Disable SSL verification (only use this for testing!)
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 120L);
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 60L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L); // Set a connection timeout


        // Perform the request
        CURLcode res = curl_easy_perform(curl);

        // Get the end time
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        double bytes_received = 0;


        if (res == CURLE_OK) {
            // After the request is complete, get the number of bytes received
            curl_easy_getinfo(curl, CURLINFO_SIZE_DOWNLOAD, &bytes_received);

            std::cout << "Bytes received: " << bytes_received << " bytes" << std::endl;

            // Get the Content-Encoding
                    char *encoding = NULL;
                    res = curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &encoding);

                    if((res == CURLE_OK) && encoding) {
                        std::cout << "Content-Encoding: " << encoding << std::endl;
                    } else {
                        std::cout << "No Content-Encoding found or error occurred." << std::endl;
                    }

        } else {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
        }

        long num_rows = 0; // Variable to store the number of rows

        if (res == CURLE_OK) {
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

            if (http_code == 200) {
                std::string response_type = "parquet"; // Assume parquet response for this example

                if (response_type == "parquet") {

                    std::string output_filename = "filtered_output.parquet";
                    std::ofstream outfile(output_filename, std::ios::binary);
                    outfile.write(response_memory.data(), GetCurrentOffset()); // Write only the used portion of the buffer
                    outfile.close();
                    std::cout << "Success: Parquet file saved as " << output_filename << std::endl;

                    // Read and print the Parquet file content
                    std::shared_ptr<arrow::io::ReadableFile> infile;
                    PARQUET_ASSIGN_OR_THROW(
                        infile, arrow::io::ReadableFile::Open(output_filename));
                    std::unique_ptr<parquet::arrow::FileReader> reader;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

                    std::shared_ptr<arrow::Table> table;
                    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

                    // Get the total number of rows in the table
                    num_rows = table->num_rows();
                    std::cout << "Total number of rows in the filtered table: " << num_rows << std::endl;

                } else if (response_type == "json") {

                    std::string json_response(response_memory.begin(), response_memory.begin() + GetCurrentOffset());
                    std::cout << "Received JSON response: " << json_response << std::endl;

                    // Parse the JSON response
                    /*try {
                        json parsed_json = json::parse(json_response);

                        // Check if "row_count" exists and is not null
                        if (parsed_json.contains("row_count") && !parsed_json["row_count"].is_null()) {
                            num_rows = parsed_json["row_count"].get<long>();
                        } else {
                            std::cerr << "Error: 'row_count' key is missing or null in the JSON response." << std::endl;
                        }

                        // You can also handle the "sum" similarly if needed
                        if (parsed_json.contains("sum") && !parsed_json["sum"].is_null()) {
                            double sum_value = parsed_json["sum"].get<double>();
                            std::cout << "Sum: " << sum_value << std::endl;
                        } else {
                            std::cerr << "Error: 'sum' key is missing or null in the JSON response." << std::endl;
                        }

                        std::cout << "Total number of rows in the filtered table: " << num_rows << std::endl;

                    } catch (const json::exception& e) {
                        std::cerr << "JSON parsing error: " << e.what() << std::endl;
                    }*/
                }

            } else {
                std::cerr << "HTTP Error: " << http_code << std::endl;
            }
        } else {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
        }

        std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;

   


// Write the timing information to the file
if (isFirstQuery) {
    timingFile.open("../timing_results_parquet.txt");
    if (timingFile.is_open()) {
        timingFile << "Query #\t" << queryIndex << "\n";
        timingFile << "Total\t" << std::fixed << std::setprecision(3) << elapsed_seconds.count() << "\n";
        timingFile << "Rows\t" << num_rows << "\n";
        timingFile << "Bytes\t" << std::fixed << std::setprecision(0) << bytes_received << " bytes\n";
        timingFile.close();
        isFirstQuery = false;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
} else {
    std::ifstream inputFile("../timing_results_parquet.txt");
    std::string line;
    std::vector<std::string> lines;

    if (inputFile.is_open()) {
        while (std::getline(inputFile, line)) {
            lines.push_back(line);
        }
        inputFile.close();

        // Reopen the file in write mode to overwrite it with new data
        timingFile.open("../timing_results_parquet.txt");
        if (timingFile.is_open()) {
            // Modify the appropriate lines to add the new data
            if (lines.size() >= 4) {
                lines[0] += "\t" + std::to_string(queryIndex);  // Append query index
                lines[1] += "\t" + std::to_string(elapsed_seconds.count());  // Append time
                lines[2] += "\t" + std::to_string(num_rows);  // Append number of rows
                lines[3] += "\t" + std::to_string(static_cast<long>(bytes_received));  // Append bytes
            }

            // Write all lines back to the file
            for (const auto& modified_line : lines) {
                timingFile << modified_line << "\n";
            }
            timingFile.close();
        } else {
            std::cerr << "Unable to open file for appending." << std::endl;
        }
    } else {
        std::cerr << "Unable to open file for reading." << std::endl;
    }
}

queryIndex++; // Increment the query index


    }

    // Clean up
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    std::cout << "All queries processed." << std::endl;

    return 0;
}
