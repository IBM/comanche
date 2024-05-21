#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm> // For std::transform
#include <cctype>    // For std::tolower
#include <locale>    // For std::locale
#include <sys/mman.h>
#include <utils.h>
#include <pistache/endpoint.h>
#include <iostream>
#include <vector>
#include <arrow/table.h> 
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/statistics.h>
#include <chrono>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h> // Include this header
#include "/home/ubuntu/json/include/nlohmann/json.hpp"
#include "SQLParser.h"

extern "C" {
    int encrypt_buffer(char* data, size_t size);
    uint8_t* decrypt_buffer(char* file_data, size_t file_size, size_t* output_size);
}

size_t header_callback(char *buffer, size_t size, size_t nitems, void *userdata) {
    std::string header(buffer, size * nitems);
    std::locale loc;  // Use the C++ locale to handle characters properly

    // Transform header to lowercase
    std::transform(header.begin(), header.end(), header.begin(),
                   [&loc](char c) { return std::tolower(c, loc); }); // Using locale

    // Find and parse the Content-Length header
    std::string content_length_key = "content-length: ";
    auto pos = header.find(content_length_key);
    if (pos != std::string::npos) {
        std::string content_length_str = header.substr(pos + content_length_key.length());
        *static_cast<size_t*>(userdata) = std::stoll(content_length_str);
    }
    return nitems * size;
}

bool GetContentSize(const std::string& url, size_t& content_size) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "CURL initialization failed." << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &content_size);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // Perform a HEAD request

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }

    return true;
}


// Callback function for writing data received from the server into memory
size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto& memory = *static_cast<std::vector<char>*>(userp);
    size_t totalSize = size * nmemb;
    static size_t currentOffset = 0; // Maintains the current offset where data is to be written

    // Check if the incoming data fits in the remaining buffer space
    /*if (currentOffset + totalSize > memory.size()) {
        std::cerr << "Buffer overflow detected: incoming data exceeds allocated buffer size." << std::endl;
        return 0; // Return 0 to signal an error to libcurl and stop the transfer
    }*/

    // Copy the received data into the vector at the current offset
    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, memory.begin() + currentOffset);
    currentOffset += totalSize; // Update the offset
    //std::cout << "Received " << totalSize << " bytes this call." << std::endl; // Print the amount of data received in this chunk

    return totalSize;
}



// Function to download a file using HTTP GET without saving it to memory
bool DownloadFileAsync(const std::string& url, std::vector<char>& data) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl initialization failed" << std::endl;
        return false;
    }
    //data.resize(746619286);

    // Set URL and other options
    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 1024*1024*2L); 
    
   

    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to fetch data: " << elapsed.count() << " seconds." << std::endl;

    // Check for errors
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    // Clean up
    curl_easy_cleanup(curl);
    return true;
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    struct timeval start, end, start_filter, end_filter;
    double time_taken = 0;

    std::string url = "http://10.10.10.18/encrypted_600000.parquet"; // Change to your actual URL
    //std::string sqlExpression = "SELECT * FROM s3object WHERE ID < 120";
    std::string sqlExpression = "SELECT * FROM s3object WHERE Age > 60";

    size_t content_size = 0;

    gettimeofday(&start, NULL);
    
    if (GetContentSize(url, content_size)) {

        gettimeofday(&end, NULL);

        time_taken = (end.tv_sec - start.tv_sec) * 1e6;
        time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
        printf("Get header size time : %.6f seconds from main\n", time_taken);

        std::cout << "Content size: " << content_size << " bytes." << std::endl;
        std::vector<char> memoryData(content_size);  // Initialize vector with the content size
        size_t output_size = 0;
        uint8_t* decrypted_data = NULL;


        if (DownloadFileAsync(url, memoryData)) {
            std::cout << "Data fetched successfully. Size: " << memoryData.size() << " bytes." << std::endl;

            if (!memoryData.empty()) {

                gettimeofday(&start, NULL);

                decrypted_data = decrypt_buffer(memoryData.data(), memoryData.size(), &output_size);

                gettimeofday(&end, NULL);

                time_taken = (end.tv_sec - start.tv_sec) * 1e6;
                time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
                printf("Decryption time taken: %.6f seconds from main\n", time_taken);


                gettimeofday(&start_filter, NULL);



                if (decrypted_data) {
                    std::cerr << "Decryption successfull " << output_size <<  std::endl;

                    auto start_total = std::chrono::high_resolution_clock::now();


                    //Filter

                    auto start_stream = std::chrono::high_resolution_clock::now();

                    auto arrowBuffer = arrow::Buffer::Wrap(decrypted_data, output_size);
                    auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);
                    std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                    auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);

                    auto end_stream = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> stream_duration = end_stream - start_stream;
                    std::cout << "Time to read stream: " << stream_duration.count() << " seconds" << std::endl;

                    
                        //Parse SQL
                    auto start_sql = std::chrono::high_resolution_clock::now();
                    std::vector<Token> tokens = SQLParser::parse(sqlExpression);

                    // Example: Analyze tokens and construct a filter (Basic and specific case handling)
                    std::string columnName;
                    std::string operatorSymbol;
                    std::string value;

                    for (const auto& token : tokens) {
                        if (token.type == TokenType::COLUMN) {
                            columnName = token.value;
                        } else if (token.type == TokenType::OPERATOR) {
                            operatorSymbol = token.value;
                        } else if (token.type == TokenType::LITERAL) {
                            value = token.value;
                        }
                // Extend with more complex logic as needed
                    }

                        
                    auto field_ref = arrow::compute::field_ref(columnName);
                    int l_value = std::stoi(value);
                    auto literal_value = arrow::compute::literal(l_value);

                    arrow::compute::Expression filter_expression;

                    // Build expression based on the operator
                    if (operatorSymbol == "=") {
                        filter_expression = arrow::compute::equal(field_ref,literal_value);
                    } else if (operatorSymbol == ">") {
                        filter_expression = arrow::compute::greater(field_ref,literal_value);
                    } else if (operatorSymbol == ">=") {
                        filter_expression = arrow::compute::greater_equal(field_ref,literal_value);
                    } else if (operatorSymbol == "<") {
                        filter_expression = arrow::compute::less(field_ref, literal_value);
                    } else if (operatorSymbol == "<=") {
                        filter_expression = arrow::compute::less_equal(field_ref,literal_value);
                    } else if (operatorSymbol == "!=") {
                        filter_expression = arrow::compute::not_equal(field_ref, literal_value);
                    }

                    auto end_sql = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> sql_duration = end_sql - start_sql;
                    std::cout << "Time to sql parse and create filter expression: " << sql_duration.count() << " seconds" << std::endl; 


        /////////////////////////////////////////////////
                    
                    auto start_stats = std::chrono::high_resolution_clock::now();

                    // Ensure there is at least one row group
                    if (arrowReader->num_row_groups() == 0) {
                        std::cerr << "No row groups found in the Parquet file." << std::endl;
                        return -1;
                    }

                    std::cout << "Number of row groups: " << arrowReader->num_row_groups()  << std::endl; 
              
             

                    // Find schema and find ID of parsed column
                    std::shared_ptr<arrow::Schema> schema;
                    arrowReader->GetSchema(&schema);
                    int column_index = -1;
                    for (int i = 0; i < schema->num_fields(); ++i) {
                        if (schema->field(i)->name() == columnName) {
                            column_index = i;
                            break;
                        }
                    }

                    if (column_index == -1) {
                        std::cerr << "Column  not found in the schema." << std::endl;
                        return -1;
                    }

                    std::vector<int> matching_row_groups;

                    for (int row_group_index = 0; row_group_index < arrowReader->num_row_groups(); ++row_group_index) {
                        auto metadata = arrowReader->parquet_reader()->metadata();
                        auto row_group_metadata = metadata->RowGroup(row_group_index);
                        auto column_metadata = row_group_metadata->ColumnChunk(column_index);
        
                        if (column_metadata->is_stats_set()) {
                            auto stats = column_metadata->statistics();
                            if (stats->HasMinMax()) {
                            // Assuming the ID column is of integer type; adjust the type as necessary
                            // Need to check the type of filter columns
                            int64_t min_value = static_cast<const parquet::Int64Statistics*>(stats.get())->min();
                            int64_t max_value = static_cast<const parquet::Int64Statistics*>(stats.get())->max();

                                if (min_value <= l_value && max_value >= l_value) {
                                    matching_row_groups.push_back(row_group_index);
                                    //std::cout << "Matching Row Group: " << row_group_index << std::endl;  // Print matching row group index
                                }
                            }
                        }
                    }

                    auto end_stats = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> stats_duration = end_stats - start_stats;
                    std::cout << "Time to read statistics and find row groups: " << stats_duration.count() << " seconds" << std::endl; 
    ///////////////////////////////////////////////////////////

                    auto start_table = std::chrono::high_resolution_clock::now();

                    std::shared_ptr<arrow::Table> concatenated_table;

                    if (!matching_row_groups.empty()) {


                        std::vector<std::shared_ptr<arrow::Table>> tables;
                        for (int row_group_index : matching_row_groups) {
                            std::shared_ptr<arrow::Table> table;
                            auto status = arrowReader->ReadRowGroup(row_group_index, &table);
                            if (!status.ok()) {
                                std::cerr << "Error reading Arrow table from RowGroup " << row_group_index << ": " << status.ToString() << std::endl;
                                continue;
                            }
                            tables.push_back(table);
                        }

                        // Assuming you want to concatenate all matching tables into a single table
                        // Assuming 'tables' is a std::vector<std::shared_ptr<arrow::Table>> containing your tables
                        arrow::Result<std::shared_ptr<arrow::Table>> concatenated_table_result = arrow::ConcatenateTables(tables);

                        if (!concatenated_table_result.ok()) {
                            // Handle error
                            std::cerr << "Failed to concatenate tables: " << concatenated_table_result.status() << std::endl;
                            return -1;
                        }

                        concatenated_table = *concatenated_table_result;


                    }else{

                        std::cout << "Matching Row Group Empty" << std::endl; 

                        auto t_status = arrowReader->ReadTable(&concatenated_table);
                        if (!t_status.ok()) {
                            std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
                            return -1;
                        }


                    }

                    auto end_table = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> table_duration = end_table - start_table;
                    std::cout << "Time to read table: " << table_duration.count() << " seconds" << std::endl;  
    ////////////////////////////////////////////////////

                    auto start_filter = std::chrono::high_resolution_clock::now();
                    auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(concatenated_table);

                    // 2: Build ScannerOptions for a Scanner to do a basic filter operation
                    auto options = std::make_shared<arrow::dataset::ScanOptions>();


                    // Build the Scanner
                    auto builder = arrow::dataset::ScannerBuilder(dataset);

                    
                    // Set the filter
                    arrow::Status build_status = builder.Filter(filter_expression);
                    if (!build_status.ok()) {
                        std::cerr << "Failed to apply filter: " << status.ToString() << std::endl;
                        return -1;
                    }

                    auto scanner = builder.Finish();

                         

                    // Perform the Scan and retrieve filtered result as Table
                    //this is the acyual filtering step
                    //this takes  time, config og scan builder is fast
                    auto result_table = scanner.ValueOrDie()->ToTable();

                    //std::cout << "Table Data:\n" << result_table.ValueUnsafe()->ToString() << std::endl;

                    auto end_filter = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> filter_duration = end_filter - start_filter;
                    std::cout << "Time to wrap in dataset and filter: " << filter_duration.count() << " seconds" << std::endl; 
              
             
                    auto end_total = std::chrono::high_resolution_clock::now();
                    auto t_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
                    printf("Time measured: %.3f seconds.\n", t_duration.count() * 1e-3);


                } else {
                    std::cout << "Decryption failed." << std::endl;
                }

                gettimeofday(&end_filter, NULL);

                free(decrypted_data);


                time_taken = (end_filter.tv_sec - start_filter.tv_sec) * 1e6;
                time_taken = (time_taken + (end_filter.tv_usec - start_filter.tv_usec)) * 1e-6;
                printf("Filtering taken: %.6f seconds from main\n", time_taken);
            }
        } else {
            std::cerr << "Data fetch failed" << std::endl;
        }
    } else {
        std::cerr << "Failed to retrieve content size." << std::endl;
    }

    curl_global_cleanup();
    return 0;
}