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
#include <pistache/endpoint.h>


using namespace Pistache;
using json = nlohmann::json;



extern "C" {
    int encrypt_buffer(char* data, size_t size);
    uint8_t* decrypt_buffer(char* file_data, size_t file_size, size_t* output_size);
        // Initialize cryptographic or any other necessary resources
    void init_crypto_resources();
        // Initialize cryptographic or any other necessary resources
    void destroy_crypto_resources();
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

// Helper function to encapsulate the static offset
static size_t& GetCurrentOffset() {
    static size_t currentOffset = 0;  // This maintains the offset
    return currentOffset;
}



void ResetWriteMemoryCallbackOffset() {
    static size_t& currentOffset = GetCurrentOffset();  // Get reference to the static offset
    currentOffset = 0;  // Reset to zero
}



// Callback function for writing data received from the server into memory
size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto& memory = *static_cast<std::vector<char>*>(userp);
    size_t totalSize = size * nmemb;
    size_t& currentOffset = GetCurrentOffset(); // Maintains the current offset where data is to be written


    // Copy the received data into the vector at the current offset
    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, memory.begin() + currentOffset);
    currentOffset += totalSize; // Update the offset
  
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

struct FilterHandler : public Http::Handler {
    HTTP_PROTOTYPE(FilterHandler)

    void onRequest(const Http::Request& req, Http::ResponseWriter response) override {

        if (req.resource() == "/data" && req.method() == Http::Method::Post) {


            json requestJson = json::parse(req.body());

            std::string bucket = requestJson["bucket"];
            std::string key = requestJson["key"];
            std::string sqlExpression = requestJson["sql"];


           

            struct timeval start, end, start_filter, end_filter, start_f, end_f;
            double time_taken = 0;

            std::string url = "http://10.10.10.18/encrypted_600000.parquet"; // Change to your actual URL
            //std::string sqlExpression = "SELECT * FROM s3object WHERE ID < 120";
            //std::string sqlExpression = "SELECT * FROM s3object WHERE Age > 60";

            size_t content_size = 0;

            gettimeofday(&start_f, NULL);

            

            try {

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

                    ResetWriteMemoryCallbackOffset();


                    if (DownloadFileAsync(url, memoryData)) {
                        std::cout << "Data fetched successfully. Size: " << memoryData.size() << " bytes." << std::endl;

                        if (!memoryData.empty()) {

                            gettimeofday(&start, NULL);

                      

                            decrypted_data = decrypt_buffer(memoryData.data(), memoryData.size(), &output_size);



                            gettimeofday(&end, NULL);

                            time_taken = (end.tv_sec - start.tv_sec) * 1e6;
                            time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
                            printf("Decryption time taken: %.6f seconds from main\n", time_taken);


                            



                            if (decrypted_data) {
                                std::cerr << "Decryption successfull " << output_size <<  std::endl;

                                auto start_total = std::chrono::high_resolution_clock::now();


                                //Filter

                                auto start_stream = std::chrono::high_resolution_clock::now();

                               //mlock(decrypted_data, output_size);  // Lock the memory

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

                                
                        

                                int num_row_groups = arrowReader->num_row_groups();

                                // Ensure there is at least one row group
                                if (num_row_groups == 0) {
                                    std::cerr << "No row groups found in the Parquet file." << std::endl;
                                    return;
                                }

                                std::cout << "Number of row groups: " << num_row_groups  << std::endl; 
                        
                        

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
                                    return;
                                }

                                std::vector<int> matching_row_groups;
                                bool useMatchingGroups = false;

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


                                // Check if there are any matching row groups
                                if (!matching_row_groups.empty()) {
                                    useMatchingGroups = true;
                                }

                                auto start_table = std::chrono::high_resolution_clock::now();




                                // Processing row groups based on whether there are matching row groups
                                if (useMatchingGroups) {
                                    for (int row_group_index : matching_row_groups) {

                                        std::shared_ptr<arrow::Table> table;
                                        status = arrowReader->RowGroup(row_group_index)->ReadTable(&table);

                                        // Wrap the Table in an InMemoryDataset
                                        std::shared_ptr<arrow::dataset::Dataset> dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

                                        // Build ScannerOptions for a Scanner to apply filter operation
                                        auto options = std::make_shared<arrow::dataset::ScanOptions>();

                                        // Build the Scanner
                                        auto builder = arrow::dataset::ScannerBuilder(dataset);     
                                         // Set the filter
                                        arrow::Status build_status = builder.Filter(filter_expression);

                                        auto scanner = builder.Finish();

                                        // Perform the Scan and retrieve filtered result as Table
                                        auto result_table = scanner.ValueOrDie()->ToTable();

                                        std::string filtered_result_json = result_table.ValueUnsafe()->ToString();

                                        response.send(Http::Code::Ok, filtered_result_json, MIME(Application, Json));


                                    }

                                } else {

                                    // If no specific matches, process all row groups
                                    for (int row_group_index = 0; row_group_index < arrowReader->num_row_groups(); ++row_group_index) {

                                        std::shared_ptr<arrow::Table> table;
                                        status = arrowReader->RowGroup(row_group_index)->ReadTable(&table);


                                         // Wrap the Table in an InMemoryDataset
                                        std::shared_ptr<arrow::dataset::Dataset> dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

                                        // Build ScannerOptions for a Scanner to apply filter operation
                                        auto options = std::make_shared<arrow::dataset::ScanOptions>();

                                        // Build the Scanner
                                        auto builder = arrow::dataset::ScannerBuilder(dataset);     
                                        // Set the filter
                                        arrow::Status build_status = builder.Filter(filter_expression);

                                        auto scanner = builder.Finish();

                                        // Perform the Scan and retrieve filtered result as Table
                                        auto result_table = scanner.ValueOrDie()->ToTable();

                                        std::string filtered_result_json = result_table.ValueUnsafe()->ToString();

                                        response.send(Http::Code::Ok, filtered_result_json, MIME(Application, Json));


                                    }
                                }





                                
              
                                auto end_table = std::chrono::high_resolution_clock::now();
                                std::chrono::duration<double> table_duration = end_table - start_table;
                                std::cout << "Time to filter: " << table_duration.count() << " seconds" << std::endl;  
                ////////////////////////////////////////////////////

                           
                           
                                gettimeofday(&end_f, NULL);
                                time_taken = (end_f.tv_sec - start_f.tv_sec) * 1e6;
                                time_taken = (time_taken + (end_f.tv_usec - start_f.tv_usec)) * 1e-6;
                                printf("Total time taken: %.6f seconds from main\n", time_taken);


                                // Send the JSON response
                               // response.send(Http::Code::Ok, filtered_result_json, MIME(Application, Json));


                            } else {
                                std::cout << "Decryption failed." << std::endl;
                            }

                            

                            free(decrypted_data);
                            decrypted_data = NULL;



                        }
                    } else {
                        std::cerr << "Data fetch failed" << std::endl;
                    }
                } else {
                    std::cerr << "Failed to retrieve content size." << std::endl;
                }

            } catch (const std::exception& e) {
                // Exception handling
                std::cerr << "Exception: " << e.what() << std::endl;
                response.send(Http::Code::Internal_Server_Error, "Exception occurred");
            }

    

            } else {

            response.send(Http::Code::Not_Found, "Endpoint not found");
        }
    }
};

int main() {


    curl_global_init(CURL_GLOBAL_DEFAULT);

        // Initialize cryptographic or any other necessary resources
    init_crypto_resources();

    Http::listenAndServe<FilterHandler>(Pistache::Address("*:8080"));

    destroy_crypto_resources();
    


    curl_global_cleanup();

    return 0;
}
