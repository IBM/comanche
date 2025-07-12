#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <curl/curl.h>
#include <vector>
#include <cstring>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/stream_writer.h>
#include <arrow/table.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/statistics.h>
#include <chrono>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h>
#include "SQLParser.h"
#include <condition_variable>
#include <queue>
#include <pthread.h>
#include <memory>
#include <fstream>
#include <arrow/io/memory.h>
#include <iomanip>
#include <sstream>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/buffer.h>
#include <arrow/type.h>
#include <arrow/array.h>
#include <arrow/compute/api_vector.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/core/auth/AWSCredentials.h>  // Include this for AWSCredentials
#include <sched.h>
#include <mutex>




#define MAX_BUFFER_SIZE 1048576 // 1MB buffer size
#define TAG_SIZE 12 // 12 bytes tag size
#define KEY_SIZE 32 // 256 bits
#define IV_SIZE 12 // 96 bits

std::mutex cout_mutex;
const size_t alloc_size = 1024 * 1024 * 1024; // 1 GB for memory allocations

void handleErrors(const std::string &message) {
    std::cerr << message << std::endl;
    ERR_print_errors_fp(stderr);
    abort();
}

std::mutex sync_mutex;
std::condition_variable sync_cv;
int ready_count = 0;
bool start_flag = false;


bool decryptChunk(const unsigned char *key, const unsigned char *iv, const unsigned char *ciphertext, int ciphertext_len, const unsigned char *tag, unsigned char *plaintext) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len;
    int plaintext_len;
    bool success = false;

    if (!ctx) return false;

    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) goto cleanup;
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, IV_SIZE, NULL) != 1) goto cleanup;
    if (EVP_DecryptInit_ex(ctx, NULL, NULL, key, iv) != 1) goto cleanup;

    if (EVP_DecryptUpdate(ctx, NULL, &len, NULL, 0) != 1) goto cleanup;  // No AAD

    if (EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len) != 1) goto cleanup;
    plaintext_len = len;

    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, TAG_SIZE, (void *)tag) != 1) goto cleanup;

    if (EVP_DecryptFinal_ex(ctx, plaintext + len, &len) != 1) goto cleanup;
    plaintext_len += len;
    success = true;

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    return success;
}

void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Failed to pin thread to core " << core_id << std::endl;
    }
}

std::vector<unsigned char> decryptBuffer(const std::vector<unsigned char>& encrypted_data, const unsigned char *key, const unsigned char *iv) {
    size_t bufferSize = MAX_BUFFER_SIZE - TAG_SIZE;
    std::vector<unsigned char> buffer(MAX_BUFFER_SIZE);
    std::vector<unsigned char> ciphertext(bufferSize);
    std::vector<unsigned char> tag(TAG_SIZE);
    std::vector<unsigned char> decrypted_data;

    size_t offset = 0;
    while (offset + MAX_BUFFER_SIZE <= encrypted_data.size()) {
        std::memcpy(ciphertext.data(), encrypted_data.data() + offset, bufferSize);
        std::memcpy(tag.data(), encrypted_data.data() + offset + bufferSize, tag.size());

        std::vector<unsigned char> plaintext(bufferSize);

        if (!decryptChunk(key, iv, ciphertext.data(), bufferSize, tag.data(), plaintext.data())) {
            handleErrors("Decryption failed!");
        }

        decrypted_data.insert(decrypted_data.end(), plaintext.begin(), plaintext.end());
        offset += MAX_BUFFER_SIZE;
    }

    // Handle the last chunk if it is smaller than MAX_BUFFER_SIZE
    size_t lastChunkSize = encrypted_data.size() - offset;
    if (lastChunkSize > 0) {
        if (lastChunkSize > TAG_SIZE) {
            buffer.resize(lastChunkSize);
            ciphertext.resize(lastChunkSize - TAG_SIZE);

            std::memcpy(ciphertext.data(), encrypted_data.data() + offset, lastChunkSize - TAG_SIZE);
            std::memcpy(tag.data(), encrypted_data.data() + offset + lastChunkSize - TAG_SIZE, tag.size());

            std::vector<unsigned char> plaintext(lastChunkSize - TAG_SIZE);

            if (!decryptChunk(key, iv, ciphertext.data(), lastChunkSize - TAG_SIZE, tag.data(), plaintext.data())) {
                handleErrors("Decryption failed!");
            }

            decrypted_data.insert(decrypted_data.end(), plaintext.begin(), plaintext.end());
        } else {
            handleErrors("Error: Last chunk is too small to contain valid data and tag.");
        }
    }

    return decrypted_data;
}

size_t header_callback(char *buffer, size_t size, size_t nitems, void *userdata) {
    std::string header(buffer, size * nitems);
    std::locale loc;

    // Transform header to lowercase
    std::transform(header.begin(), header.end(), header.begin(),
                   [&loc](char c) { return std::tolower(c, loc); });

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

static size_t& GetCurrentOffset() {
    static size_t currentOffset = 0;
    return currentOffset;
}

void ResetWriteMemoryCallbackOffset() {
    static size_t& currentOffset = GetCurrentOffset();
    currentOffset = 0;
}

size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto& memory = *static_cast<std::vector<char>*>(userp);
    size_t totalSize = size * nmemb;
    size_t& currentOffset = GetCurrentOffset();

    // Copy the received data into the vector at the current offset
    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, memory.begin() + currentOffset);
    currentOffset += totalSize;

    return totalSize;
}

bool DownloadFileAsync(const std::string& url, std::vector<char>& data) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl initialization failed" << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 1024*1024*2L);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    auto start = std::chrono::high_resolution_clock::now();
    CURLcode res = curl_easy_perform(curl);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to fetch data: " << elapsed.count() << " seconds." << std::endl;

    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_cleanup(curl);
    return true;
}


arrow::compute::Expression GetFilterExpression(const std::string& sqlExpression) {
    std::vector<Token> tokens = SQLParser::parse(sqlExpression);

    std::vector<arrow::compute::Expression> conditions;
    std::string columnName;
    std::string operatorSymbol;
    std::string value;
    std::string value2;

    bool processingWhereClause = false;

    try {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].type == TokenType::WHERE) {
                processingWhereClause = true;
                continue;
            }

            if (processingWhereClause) {
                if (tokens[i].type == TokenType::COLUMN) {
                    columnName = tokens[i].value;
                    if (i + 2 >= tokens.size()) {
                        std::cerr << "Error: missing operator or value after column " << columnName << std::endl;
                        throw std::invalid_argument("Invalid SQL expression: missing operator or value");
                    }
                    operatorSymbol = tokens[i + 1].value;

                    if (operatorSymbol == "BETWEEN") {
                        if (i + 4 >= tokens.size()) {
                            std::cerr << "Error: missing values for BETWEEN operator after column " << columnName << std::endl;
                            throw std::invalid_argument("Invalid SQL expression: missing values for BETWEEN operator");
                        }
                        value = tokens[i + 2].value;
                        if (tokens[i + 3].type != TokenType::AND) {
                            std::cerr << "Error: missing AND in BETWEEN clause for column " << columnName << std::endl;
                            throw std::invalid_argument("Invalid SQL expression: missing AND in BETWEEN clause");
                        }
                        value2 = tokens[i + 4].value;
                        i += 4;
                    } else {
                        value = tokens[i + 2].value;
                        i += 2;
                    }

                    // Remove quotes from the value if present
                    if (value.front() == '\'' && value.back() == '\'') {
                        value = value.substr(1, value.size() - 2);
                    }
                    if (!value2.empty() && value2.front() == '\'' && value2.back() == '\'') {
                        value2 = value2.substr(1, value2.size() - 2);
                    }

                    auto field_ref = arrow::compute::field_ref(columnName);
                    arrow::compute::Expression condition;

                    // Handle int64 and nullable int64 types for inventory columns
                    auto literal_value = arrow::compute::literal(static_cast<int64_t>(std::stoll(value)));
                    if (operatorSymbol == "=") {
                        condition = arrow::compute::equal(field_ref, literal_value);
                    } else if (operatorSymbol == ">") {
                        condition = arrow::compute::greater(field_ref, literal_value);
                    } else if (operatorSymbol == ">=") {
                        condition = arrow::compute::greater_equal(field_ref, literal_value);
                    } else if (operatorSymbol == "<") {
                        condition = arrow::compute::less(field_ref, literal_value);
                    } else if (operatorSymbol == "<=") {
                        condition = arrow::compute::less_equal(field_ref, literal_value);
                    } else if (operatorSymbol == "!=") {
                        condition = arrow::compute::not_equal(field_ref, literal_value);
                    } else if (operatorSymbol == "BETWEEN") {
                        auto literal_value2 = arrow::compute::literal(static_cast<int64_t>(std::stoll(value2)));
                        condition = arrow::compute::and_(arrow::compute::greater_equal(field_ref, literal_value),
                                                         arrow::compute::less_equal(field_ref, literal_value2));
                    }

                    conditions.push_back(condition);
                }
            }
        }

        // Combine all conditions using 'and'
        if (!conditions.empty()) {
            arrow::compute::Expression combined_condition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                combined_condition = arrow::compute::and_(combined_condition, conditions[i]);
                //std::cout << "Condition " << i << ": " << conditions[i].ToString() << std::endl;
            }
            //std::cout << "Combined condition: " << combined_condition.ToString() << std::endl;

            return combined_condition;
        } else {
            throw std::invalid_argument("No valid conditions found");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in GetFilterExpression: " << e.what() << std::endl;
        throw;
    }
}

std::vector<std::string> GetSelectColumns(const std::string& sqlExpression) {
    std::vector<Token> tokens = SQLParser::parse(sqlExpression);
    std::vector<std::string> columns;
    bool processingSelectClause = false;

    try {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].type == TokenType::SELECT) {
                processingSelectClause = true;
                continue;
            }
            if (tokens[i].type == TokenType::FROM) {
                processingSelectClause = false;
                break;
            }
            if (processingSelectClause && (tokens[i].type == TokenType::COLUMN || tokens[i].type == TokenType::UNKNOWN)) {
                columns.push_back(tokens[i].value);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in GetSelectColumns: " << e.what() << std::endl;
        throw;
    }

    // Debugging prints
    /*std::cout << "Selected columns: ";
    for (const auto& col : columns) {
        std::cout << col << " ";
    }
    std::cout << std::endl;*/

    return columns;
}

void filterDataSync(const arrow::compute::Expression& filter_expression, std::shared_ptr<arrow::io::BufferReader> bufferReader, int row_group_index, std::shared_ptr<arrow::Table>& final_result_table, const std::vector<std::string>& select_columns) {
    try {
        std::unique_ptr<parquet::arrow::FileReader> arrowReader;
        auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);

        if (!status.ok()) {
            //std::lock_guard<std::mutex> lock(response_mutex);
            return;
        }

        std::shared_ptr<parquet::arrow::FileReader> shared_arrowReader = std::shared_ptr<parquet::arrow::FileReader>(std::move(arrowReader));

        std::shared_ptr<arrow::Table> table;
        status = shared_arrowReader->RowGroup(row_group_index)->ReadTable(&table);
        if (!status.ok()) {
            //std::lock_guard<std::mutex> lock(response_mutex);
            //response->send(Http::Code::Internal_Server_Error, "Failed to read row group.");
            return;
        }

        std::shared_ptr<arrow::dataset::Dataset> dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

        auto options = std::make_shared<arrow::dataset::ScanOptions>();

        auto builder = arrow::dataset::ScannerBuilder(dataset);
        builder.Filter(filter_expression);

        // Apply projection if not aggregation
    
        builder.Project(select_columns);
        

        auto scanner_result = builder.Finish();
        if (!scanner_result.ok()) {
            //std::lock_guard<std::mutex> lock(response_mutex);
            //response->send(Http::Code::Internal_Server_Error, "Failed to build scanner.");
            return;
        }
        auto scanner = scanner_result.ValueOrDie();

        auto result_table_result = scanner->ToTable();
        if (!result_table_result.ok()) {
            //std::lock_guard<std::mutex> lock(response_mutex);
            //response->send(Http::Code::Internal_Server_Error, "Failed to convert to table.");
            return;
        }

        auto result_table = result_table_result.ValueOrDie();

        // Update the filtered row count
       

        // Skip empty tables
        if (result_table->num_rows() == 0) {
            return;
        }

        // Filter out empty rows
        std::vector<std::shared_ptr<arrow::ChunkedArray>> filtered_columns;
        for (const auto& column : result_table->columns()) {
            std::vector<std::shared_ptr<arrow::Array>> non_empty_chunks;
            for (const auto& chunk : column->chunks()) {
                if (chunk->length() > 0) {
                    non_empty_chunks.push_back(chunk);
                }
            }
            if (!non_empty_chunks.empty()) {
                filtered_columns.push_back(std::make_shared<arrow::ChunkedArray>(non_empty_chunks));
            }
        }

        result_table = arrow::Table::Make(result_table->schema(), filtered_columns);

      
    
            // Combine non-empty result tables
            //std::lock_guard<std::mutex> lock(table_mutex);
            if (final_result_table == nullptr) {
                final_result_table = result_table;
            } else {
                std::vector<std::shared_ptr<arrow::Table>> non_empty_tables;
                if (final_result_table->num_rows() > 0) {
                    non_empty_tables.push_back(final_result_table);
                }
                if (result_table->num_rows() > 0) {
                    non_empty_tables.push_back(result_table);
                }

                if (!non_empty_tables.empty()) {
                    auto concatenate_result = arrow::ConcatenateTables(non_empty_tables, arrow::ConcatenateTablesOptions::Defaults());
                    if (!concatenate_result.ok()) {
                        //std::lock_guard<std::mutex> lock(response_mutex);
                        //response->send(Http::Code::Internal_Server_Error, "Failed to concatenate tables.");
                        return;
                    }
                    final_result_table = concatenate_result.ValueOrDie();
                }
            }
        
    } catch (const std::exception& e) {
        //std::lock_guard<std::mutex> lock(response_mutex);
        //response->send(Http::Code::Internal_Server_Error, e.what());
    }
}



void run_query_thread(int thread_id, const std::string& sqlExpression, const std::string& key_file, int level) {
    pin_to_core(thread_id);

    const std::string bucket = "encrypted";

    Aws::String minioEndpointUrl = "http://10.10.10.18:9000";
    Aws::String awsAccessKey = "minioadmin";
    Aws::String awsSecretKey = "minioadmin";

    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.endpointOverride = minioEndpointUrl;
    clientConfig.scheme = Aws::Http::Scheme::HTTP;
    clientConfig.verifySSL = false;

    std::vector<char> memoryData(alloc_size);

    Aws::Auth::AWSCredentials credentials(awsAccessKey, awsSecretKey);
    Aws::S3::S3Client s3Client(credentials, clientConfig, Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);

    auto start_total = std::chrono::high_resolution_clock::now();

    try {
        Aws::S3::Model::GetObjectRequest getObjectRequest;
        getObjectRequest.SetBucket(bucket.c_str());
        getObjectRequest.SetKey(key_file.c_str());


        auto fetch_start = std::chrono::high_resolution_clock::now();
        auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
        auto fetch_end = std::chrono::high_resolution_clock::now();

        if (!getObjectOutcome.IsSuccess()) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "[Thread " << thread_id << "] S3 fetch failed: "
                      << getObjectOutcome.GetError().GetMessage() << std::endl;
            return;
        }

        auto& objectStream = getObjectOutcome.GetResult().GetBody();
        objectStream.seekg(0, std::ios::end);
        std::streamsize size = objectStream.tellg();
        objectStream.seekg(0, std::ios::beg);

        if (size <= 0) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "[Thread " << thread_id << "] Empty file." << std::endl;
            return;
        }

        memoryData.resize(static_cast<size_t>(size));
        objectStream.read(memoryData.data(), size);

        auto decrypt_start = std::chrono::high_resolution_clock::now();
        const unsigned char key[KEY_SIZE] = {0};
        const unsigned char iv[IV_SIZE] = {0};
        std::vector<unsigned char> encrypted_data(memoryData.begin(), memoryData.end());
        std::vector<unsigned char> decrypted_data = decryptBuffer(encrypted_data, key, iv);
        auto decrypt_end = std::chrono::high_resolution_clock::now();

        if (decrypted_data.empty()) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "[Thread " << thread_id << "] Decryption failed." << std::endl;
            return;
        }

        auto arrowBuffer = std::make_shared<arrow::Buffer>(decrypted_data.data(), decrypted_data.size());
        auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);

        std::unique_ptr<parquet::arrow::FileReader> arrowReader;
        PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader));

        int num_row_groups = arrowReader->num_row_groups();
        if (num_row_groups == 0) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "[Thread " << thread_id << "] No row groups." << std::endl;
            return;
        }

        auto filter_expression = GetFilterExpression(sqlExpression);
        auto select_columns = GetSelectColumns(sqlExpression);

        std::shared_ptr<arrow::Table> final_table;
        auto filter_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_row_groups; ++i) {
            filterDataSync(filter_expression, bufferReader, i, final_table, select_columns);
        }
        auto filter_end = std::chrono::high_resolution_clock::now();

        auto end_total = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(cout_mutex);
        /*std::cout << "[Thread " << thread_id << "] File: " << key_file
                  << ", Rows: " << (final_table ? final_table->num_rows() : 0)
                  << ", Fetch: " << std::chrono::duration<double>(fetch_end - fetch_start).count() << "s"
                  << ", Decrypt: " << std::chrono::duration<double>(decrypt_end - decrypt_start).count() << "s"
                  << ", Filter: " << std::chrono::duration<double>(filter_end - filter_start).count() << "s"
                  << ", Total: " << std::chrono::duration<double>(end_total - start_total).count() << "s"
                  << std::endl;*/

    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "[Thread " << thread_id << "] Exception: " << e.what() << std::endl;
    }
}


int main() {
    if (setenv("AWS_EC2_METADATA_DISABLED", "true", 1) != 0) {}

    Aws::SDKOptions options;
    Aws::InitAPI(options);

    // Read the single SQL query
    std::ifstream file("../queries.txt");
    std::string query;
    if (!std::getline(file, query)) {
        std::cerr << "Failed to read query from queries.txt" << std::endl;
        Aws::ShutdownAPI(options);
        return 1;
    }

    std::vector<int> concurrency_levels(16);
    std::iota(concurrency_levels.begin(), concurrency_levels.end(), 1);  // [1, 2, ..., 16]
//int level =9 ;
    for (int level : concurrency_levels) {
        std::vector<std::thread> threads;
        auto start_all = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < level; ++i) {
            std::string file_key = "enc_inventory_" + std::to_string(i + 1) + ".parquet";
            threads.emplace_back(run_query_thread, i, query, file_key, level);
        }

        for (auto& t : threads) t.join();

        auto end_all = std::chrono::high_resolution_clock::now();
        double total_elapsed = std::chrono::duration<double>(end_all - start_all).count();

        std::cout << "[Concurrency: " << level << "] Total time: " << total_elapsed
                  << "s, Throughput: " << level / total_elapsed << " QPS" << std::endl << std::endl;
    }

    Aws::ShutdownAPI(options);
    return 0;
}
