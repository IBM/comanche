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

// Function to write the response data to a string
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t totalSize = size * nmemb;
    s->append((char*)contents, totalSize);
    return totalSize;
}

int main() {
    // URL and payload data
    std::string url = "http://10.10.10.20:8080/data";
    std::string payload = R"({
        "bucket": "mycsvbucket",
        "key": "sampledata/dataStat_1000000.parquet",
        "sql": "SELECT * FROM s3object WHERE Age > 60"
    })";

    // Headers
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Initialize CURL
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return 1;
    }

    // Variables to store the response and timing
    std::string response_string;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Set CURL options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // Disable SSL verification (only use this for testing!)
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 120L);
curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 60L);

curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L); // Set a timeout for the entire request
curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L); // Set a connection timeout


    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    // Get the end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Check for errors
    if (res != CURLE_OK) {
        std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
    } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code == 200) {
            std::string output_filename = "filtered_output.parquet";
            std::ofstream outfile(output_filename, std::ios::binary);
            outfile.write(response_string.c_str(), response_string.size());
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

            // Printing the Parquet table
            std::cout << "Table contents:" << std::endl;
            std::cout << table->ToString() << std::endl;
        } else {
            std::cerr << "HTTP Error: " << http_code << std::endl;
        }
    }

    // Clean up
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
