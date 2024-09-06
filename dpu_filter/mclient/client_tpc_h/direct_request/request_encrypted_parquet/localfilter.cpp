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


#define MAX_BUFFER_SIZE 1048576 // 1MB buffer size
#define TAG_SIZE 12 // 12 bytes tag size
#define KEY_SIZE 32 // 256 bits
#define IV_SIZE 12 // 96 bits

void handleErrors(const std::string &message) {
    std::cerr << message << std::endl;
    ERR_print_errors_fp(stderr);
    abort();
}

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

                    if (columnName == "l_shipdate") {
                        std::istringstream ss(value);
                        std::tm tm = {};
                        ss >> std::get_time(&tm, "%Y-%m-%d");
                        if (ss.fail()) {
                            throw std::invalid_argument("Failed to parse date");
                        }
                        auto ts = std::chrono::system_clock::from_time_t(std::mktime(&tm));
                        auto timestamp_scalar = std::make_shared<arrow::TimestampScalar>(ts.time_since_epoch().count(), arrow::timestamp(arrow::TimeUnit::NANO));
                        auto literal_value = arrow::compute::literal(timestamp_scalar);

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
                        }
                    } else if (columnName == "l_quantity" || columnName == "l_extendedprice" || columnName == "l_discount") {
                        auto literal_value = arrow::compute::literal(std::stod(value));

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
                            auto literal_value2 = arrow::compute::literal(std::stod(value2));
                            condition = arrow::compute::and_(arrow::compute::greater_equal(field_ref, literal_value),
                                                             arrow::compute::less_equal(field_ref, literal_value2));
                        }
                    } else {
                        auto literal_value = (columnName == "l_orderkey" || columnName == "l_partkey" || columnName == "l_suppkey" || columnName == "l_linenumber" || columnName == "__index_level_0__")
                                            ? arrow::compute::literal(static_cast<int64_t>(std::stoll(value)))
                                            : arrow::compute::literal(value);

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
                        }
                    }

                    conditions.push_back(condition);
                }
            }
        }

        if (!conditions.empty()) {
            arrow::compute::Expression combined_condition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                combined_condition = arrow::compute::and_(combined_condition, conditions[i]);
            }
            return combined_condition;
        } else {
            throw std::invalid_argument("No valid conditions found");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in GetFilterExpression: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    const std::string url = "http://10.10.10.18/parquet_files/enc_lineitem.parquet";
    const std::string sqlExpression = "SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24000";

    struct timeval start, end, start_f, end_f;
    double time_taken = 0;

    size_t content_size = 0;

    // Define the key and iv variables in the main function
const unsigned char key[KEY_SIZE] = {0}; // 256-bit key (all zeros)
const unsigned char iv[IV_SIZE] = {0};   // 96-bit IV (all zeros)


    if (GetContentSize(url, content_size)) {
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

                        std::vector<unsigned char> encrypted_data(memoryData.begin(), memoryData.end());
                        std::vector<unsigned char> decrypted_data = decryptBuffer(encrypted_data, key, iv);

                        gettimeofday(&end, NULL);

                        time_taken = (end.tv_sec - start.tv_sec) * 1e6;
                        time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
                        printf("Decryption time taken: %.6f seconds from main\n", time_taken);

                        if (!decrypted_data.empty()) {
                            std::cerr << "Decryption successful. " << decrypted_data.size() << " bytes decrypted." << std::endl;

                            auto start_total = std::chrono::high_resolution_clock::now();

                            auto start_stream = std::chrono::high_resolution_clock::now();

                            auto arrowBuffer = std::make_shared<arrow::Buffer>(decrypted_data.data(), decrypted_data.size());
                            auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);
                            std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                            auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);

                            auto end_stream = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> stream_duration = end_stream - start_stream;
                            std::cout << "Time to read stream: " << stream_duration.count() << " seconds" << std::endl;

                            auto start_sql = std::chrono::high_resolution_clock::now();
                            auto filter_expression = GetFilterExpression(sqlExpression);
                            auto end_sql = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> sql_duration = end_sql - start_sql;
                            std::cout << "Time to sql parse and create filter expression: " << sql_duration.count() << " seconds" << std::endl;

                            auto start_table = std::chrono::high_resolution_clock::now();

                            std::shared_ptr<arrow::Table> table;
                            status = arrowReader->ReadTable(&table);

                            std::shared_ptr<arrow::dataset::Dataset> dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

                            auto options = std::make_shared<arrow::dataset::ScanOptions>();

                            auto builder = arrow::dataset::ScannerBuilder(dataset);
                            builder.Filter(filter_expression);

                            auto scanner_result = builder.Finish();
                            if (!scanner_result.ok()) {
                                std::cerr << "Failed to build scanner." << std::endl;
                                return 1;
                            }
                            auto scanner = scanner_result.ValueOrDie();

                            auto result_table_result = scanner->ToTable();
                            if (!result_table_result.ok()) {
                                std::cerr << "Failed to convert to table." << std::endl;
                                return 1;
                            }

                            auto result_table = result_table_result.ValueOrDie();

                            auto extendedprice = result_table->GetColumnByName("l_extendedprice");
                            auto discount = result_table->GetColumnByName("l_discount");

                            auto multiply_result = arrow::compute::CallFunction("multiply", {extendedprice->chunk(0), discount->chunk(0)});
                            if (!multiply_result.ok()) {
                                std::cerr << "Failed to multiply columns." << std::endl;
                                return 1;
                            }
                            auto revenue_column = multiply_result.ValueOrDie().make_array();

                            auto sum_result = arrow::compute::Sum(revenue_column);
                            if (!sum_result.ok()) {
                                std::cerr << "Failed to compute sum." << std::endl;
                                return 1;
                            }
                            auto revenue = std::static_pointer_cast<arrow::DoubleScalar>(sum_result.ValueOrDie().scalar())->value;

                            auto revenue_array = std::make_shared<arrow::DoubleArray>(1, arrow::Buffer::Wrap(std::vector<double>{revenue}));
                            auto schema = arrow::schema({arrow::field("revenue", arrow::float64())});
                            auto result_table_with_revenue = arrow::Table::Make(schema, {revenue_array});

                            std::cout << "Final table contents:" << std::endl;
                            std::cout << result_table_with_revenue->ToString() << std::endl;

                            auto end_table = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> table_duration = end_table - start_table;
                            std::cout << "Time to filter: " << table_duration.count() << " seconds" << std::endl;

                            gettimeofday(&end_f, NULL);
                            time_taken = (end_f.tv_sec - start_f.tv_sec) * 1e6;
                            time_taken = (time_taken + (end_f.tv_usec - start_f.tv_usec)) * 1e-6;
                            printf("Total time taken: %.6f seconds from main\n", time_taken);

                        } else {
                            std::cout << "Decryption failed." << std::endl;
                        }
                    }
                } else {
                    std::cerr << "Data fetch failed" << std::endl;
                }
            } else {
                std::cerr << "Failed to retrieve content size." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    curl_global_cleanup();
    return 0;
}
