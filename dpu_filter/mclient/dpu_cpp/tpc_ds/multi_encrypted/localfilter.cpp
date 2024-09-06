#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <locale>
#include <sys/mman.h>
#include <utils.h>
#include <pistache/endpoint.h>
#include <vector>
#include <arrow/table.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/statistics.h>
#include <parquet/exception.h>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h>
#include "nlohmann/json.hpp"
#include "SQLParser.h"
#include <fstream>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <arrow/io/memory.h>
#include <iomanip>
#include <sstream>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/buffer.h>
#include <arrow/type.h>
#include <arrow/array.h>
#include <parquet/arrow/writer.h>
#include <arrow/compute/api_vector.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#include <arrow/io/api.h>
#include <parquet/file_writer.h>

using namespace Pistache;
using json = nlohmann::json;

std::mutex decryption_mutex;
std::mutex response_mutex;
std::mutex table_mutex;
std::mutex write_mutex;

const size_t alloc_size = 1024 * 1024 * 1024; // 1 GB for memory allocations
const size_t max_rowgroup_size = 512 * 1024 * 1024;

std::chrono::duration<double, std::milli> total_encrypt_time(0);

extern "C" {
    // Decrypt functions
    uint8_t* decrypt_buffer(char* file_data, size_t file_size, size_t* output_size, uint8_t* dst_buffer);
    void init_crypto_resources();
    void destroy_crypto_resources();
    uint8_t* prep_doca_buffer_dst(const size_t file_size);
    uint8_t* prep_doca_buffer_src(const size_t file_size, char* file_data);
    void stop_mmap();

    // Encrypt functions
    uint8_t* encrypt_buffer(char* file_data, size_t file_size, size_t* output_size, uint8_t* dst_buffer);
    void enc_init_crypto_resources();
    void enc_destroy_crypto_resources();
    uint8_t* enc_prep_doca_buffer_dst(const size_t file_size);
    uint8_t* enc_prep_doca_buffer_src(const size_t file_size, char* file_data);
    void enc_stop_mmap();
}

class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type>;
    
    size_t getNumThreads() const { return num_threads; }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    size_t num_threads;

    void set_affinity(std::thread::native_handle_type handle, int core_id);
};

ThreadPool::ThreadPool(size_t num_threads) : stop(false), num_threads(num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this, i] {
            set_affinity(pthread_self(), i);
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }
}

void ThreadPool::set_affinity(std::thread::native_handle_type handle, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset);
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
        worker.join();
}

template<class F>
auto ThreadPool::enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
    using return_type = typename std::result_of<F()>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

// Single global thread pool
ThreadPool thread_pool(std::thread::hardware_concurrency());

static size_t& GetCurrentOffset() {
    static size_t currentOffset = 0;
    return currentOffset;
}

void ResetWriteMemoryCallbackOffset() {
    static size_t& currentOffset = GetCurrentOffset();
    currentOffset = 0;
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

void decryptAndProcessData(char* data, size_t size, size_t* output_size, uint8_t* dst_buffer) {
    std::lock_guard<std::mutex> lock(decryption_mutex);
    uint8_t* decrypted_data = decrypt_buffer(data, size, output_size, dst_buffer);
    if (!decrypted_data) {
        std::cerr << "Decryption failed" << std::endl;
        throw std::runtime_error("Decryption failed");
    }
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
                std::cout << "Condition " << i << ": " << conditions[i].ToString() << std::endl;
            }
            std::cout << "Combined condition: " << combined_condition.ToString() << std::endl;

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
    std::cout << "Selected columns: ";
    for (const auto& col : columns) {
        std::cout << col << " ";
    }
    std::cout << std::endl;

    return columns;
}

std::vector<int> GetMatchingRowGroups(const std::unique_ptr<parquet::arrow::FileReader>& arrowReader, const std::string& columnName, int l_value) {
    std::vector<int> matching_row_groups;
    std::shared_ptr<arrow::Schema> schema;
    arrowReader->GetSchema(&schema);

    int column_index = schema->GetFieldIndex(columnName);
    if (column_index == -1) {
        throw std::runtime_error("Column not found in the schema.");
    }

    for (int row_group_index = 0; row_group_index < arrowReader->num_row_groups(); ++row_group_index) {
        auto metadata = arrowReader->parquet_reader()->metadata();
        auto row_group_metadata = metadata->RowGroup(row_group_index);
        auto column_metadata = row_group_metadata->ColumnChunk(column_index);

        if (column_metadata->is_stats_set()) {
            auto stats = column_metadata->statistics();
            if (stats->HasMinMax()) {
                int64_t min_value = static_cast<const parquet::Int64Statistics*>(stats.get())->min();
                int64_t max_value = static_cast<const parquet::Int64Statistics*>(stats.get())->max();

                if (min_value <= l_value && max_value >= l_value) {
                    matching_row_groups.push_back(row_group_index);
                }
            }
        }
    }

    return matching_row_groups;
}

void filterDataSync(const arrow::compute::Expression& filter_expression, std::shared_ptr<arrow::io::BufferReader> bufferReader, int row_group_index, std::shared_ptr<Http::ResponseWriter> response, nlohmann::json& aggregation_result, std::shared_ptr<arrow::Table>& final_result_table, const std::vector<std::string>& select_columns, bool is_aggregation) {
    try {
        std::unique_ptr<parquet::arrow::FileReader> arrowReader;
        auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);

        if (!status.ok()) {
            std::lock_guard<std::mutex> lock(response_mutex);
            response->send(Http::Code::Internal_Server_Error, "Failed to open file");
            return;
        }

        std::shared_ptr<parquet::arrow::FileReader> shared_arrowReader = std::shared_ptr<parquet::arrow::FileReader>(std::move(arrowReader));

        std::shared_ptr<arrow::Table> table;
        status = shared_arrowReader->RowGroup(row_group_index)->ReadTable(&table);
        if (!status.ok()) {
            std::lock_guard<std::mutex> lock(response_mutex);
            response->send(Http::Code::Internal_Server_Error, "Failed to read row group.");
            return;
        }

        std::shared_ptr<arrow::dataset::Dataset> dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

        auto options = std::make_shared<arrow::dataset::ScanOptions>();

        auto builder = arrow::dataset::ScannerBuilder(dataset);
        builder.Filter(filter_expression);

        // Apply projection if not aggregation
        if (!is_aggregation) {
            builder.Project(select_columns);
        }

        auto scanner_result = builder.Finish();
        if (!scanner_result.ok()) {
            std::lock_guard<std::mutex> lock(response_mutex);
            response->send(Http::Code::Internal_Server_Error, "Failed to build scanner.");
            return;
        }
        auto scanner = scanner_result.ValueOrDie();

        auto result_table_result = scanner->ToTable();
        if (!result_table_result.ok()) {
            std::lock_guard<std::mutex> lock(response_mutex);
            response->send(Http::Code::Internal_Server_Error, "Failed to convert to table.");
            return;
        }

        auto result_table = result_table_result.ValueOrDie();

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

        if (is_aggregation) {
            // Handle aggregation (SUM and COUNT)
            auto quantity_column = result_table->GetColumnByName("inv_quantity_on_hand");

            auto count_result = arrow::compute::Count(quantity_column);
            if (!count_result.ok()) {
                std::lock_guard<std::mutex> lock(response_mutex);
                response->send(Http::Code::Internal_Server_Error, "Failed to compute count.");
                return;
            }

            auto sum_result = arrow::compute::Sum(quantity_column);
            if (!sum_result.ok()) {
                std::lock_guard<std::mutex> lock(response_mutex);
                response->send(Http::Code::Internal_Server_Error, "Failed to compute sum.");
                return;
            }

            auto count_value = std::static_pointer_cast<arrow::Int64Scalar>(count_result.ValueOrDie().scalar())->value;
            auto sum_value = std::static_pointer_cast<arrow::DoubleScalar>(sum_result.ValueOrDie().scalar())->value;

            // Accumulate results in aggregation_result JSON
            if (aggregation_result.contains("count")) {
                aggregation_result["count"] = aggregation_result["count"].get<int64_t>() + count_value;
            } else {
                aggregation_result["count"] = count_value;
            }

            if (aggregation_result.contains("sum")) {
                aggregation_result["sum"] = aggregation_result["sum"].get<double>() + sum_value;
            } else {
                aggregation_result["sum"] = sum_value;
            }
        } else {
            // Combine non-empty result tables
            std::lock_guard<std::mutex> lock(table_mutex);
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
                        std::lock_guard<std::mutex> lock(response_mutex);
                        response->send(Http::Code::Internal_Server_Error, "Failed to concatenate tables.");
                        return;
                    }
                    final_result_table = concatenate_result.ValueOrDie();
                }
            }
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(response_mutex);
        response->send(Http::Code::Internal_Server_Error, e.what());
    }
}

void prefetchDataBuffer(std::vector<char>& buffer) {
    for (size_t i = 0; i < buffer.size(); i += 4096) {
        volatile char tmp = buffer[i];
    }
}

bool DownloadS3Range(const Aws::S3::S3Client& s3Client, const std::string& bucket, const std::string& key, size_t start, size_t end, std::vector<char>& buffer) {
    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.SetBucket(bucket.c_str());
    getObjectRequest.SetKey(key.c_str());
    getObjectRequest.SetRange(("bytes=" + std::to_string(start) + "-" + std::to_string(end)).c_str());

    auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
    if (getObjectOutcome.IsSuccess()) {
        auto& objectStream = getObjectOutcome.GetResult().GetBody();
        objectStream.read(buffer.data() + start, end - start + 1);

        // Prefetch the data buffer to improve cache performance
        for (size_t i = start; i <= end; i += 4096) {
            volatile char tmp = buffer[i];
        }

        return true;
    } else {
        std::cerr << "Failed to get range " << start << "-" << end << ": " << getObjectOutcome.GetError().GetMessage() << std::endl;
        return false;
    }
}



class FilterHandler : public Http::Handler {
    HTTP_PROTOTYPE(FilterHandler)

public:
    FilterHandler(std::vector<char>& data, uint8_t* dst_buf, uint8_t* final_buf, std::vector<char>& decrypt_buf)
        : data(data), dst_buffer(dst_buf), final_buffer(final_buf), decrypt_buffer(decrypt_buf) {}

    void onRequest(const Http::Request& req, Http::ResponseWriter response) override {

            static int queryIndex = 1; // Keep track of the query number
            static bool isFirstQuery = true; // Track if this is the first query to write headers
            std::ofstream outputFile;

            // Declare the timing variables at the start of the method
            std::chrono::duration<double> fetchTime;
            std::chrono::duration<double> decryptTime;
            std::chrono::duration<double> filterTime;
            std::chrono::duration<double> writeTime;
            std::chrono::duration<double> totalTime;    
     
     
     
        if (req.resource() == "/data" && req.method() == Http::Method::Post) {
            json requestJson = json::parse(req.body());

            std::string bucket = requestJson["bucket"];
            std::string key = requestJson["key"];
            std::string sqlExpression = requestJson["sql"];
            std::string output = "parquet";   //by default parquet

            // Check if the "output" field exists in the requestJson
            if (requestJson.contains("output")) {
                output = requestJson["output"];
            }

            Aws::SDKOptions options;
            Aws::InitAPI(options);

            Aws::String minioEndpointUrl = "http://10.10.10.18:9000";
            Aws::String awsAccessKey = "minioadmin";
            Aws::String awsSecretKey = "minioadmin";

            Aws::Client::ClientConfiguration clientConfig;
            clientConfig.endpointOverride = minioEndpointUrl;
            clientConfig.scheme = Aws::Http::Scheme::HTTP;
            clientConfig.verifySSL = false;

            Aws::Auth::AWSCredentials credentials(awsAccessKey, awsSecretKey);

            Aws::S3::S3Client s3Client(credentials, clientConfig, Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);

            auto start_f = std::chrono::high_resolution_clock::now();

            try {
                Aws::S3::Model::HeadObjectRequest headObjectRequest;
                headObjectRequest.SetBucket(bucket.c_str());
                headObjectRequest.SetKey(key.c_str());

                auto headObjectOutcome = s3Client.HeadObject(headObjectRequest);
                if (headObjectOutcome.IsSuccess()) {
                    auto objectSize = headObjectOutcome.GetResult().GetContentLength();

                    size_t part_size = 8 * 1024 * 1024; // 8 MB
                    size_t num_parts = (objectSize + part_size - 1) / part_size;
                    data.resize(objectSize);

                    std::vector<std::future<bool>> futures;
                    for (size_t i = 0; i < num_parts; ++i) {
                        size_t start = i * part_size;
                        size_t end = std::min(start + part_size - 1, static_cast<size_t>(objectSize - 1));
                        futures.push_back(thread_pool.enqueue([&, start, end] {
                            return DownloadS3Range(s3Client, bucket, key, start, end, data);
                        }));
                    }

                    bool success = true;
                    for (auto& future : futures) {
                        if (!future.get()) {
                            success = false;
                        }
                    }

                    if (success) {
                        auto end_s3_fetch = std::chrono::high_resolution_clock::now();
                        fetchTime = end_s3_fetch - start_f;
                        std::cout << "Time to fetch object from S3: " << fetchTime.count() << " seconds" << std::endl;

                        auto start_decrypt = std::chrono::high_resolution_clock::now();

                        size_t output_size = 0;
                        decryptAndProcessData(data.data(), data.size(), &output_size, dst_buffer);

                        auto end_decrypt = std::chrono::high_resolution_clock::now();
                        decryptTime = end_decrypt - start_decrypt;
                        std::cout << "Decryption time taken: " << decryptTime.count() << " seconds" << std::endl;

                        auto start_filter = std::chrono::high_resolution_clock::now();

                        auto arrowBuffer = arrow::Buffer::Wrap(dst_buffer, output_size);
                        auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);
                        std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                        auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);
                        if (!status.ok()) {
                            response.send(Http::Code::Internal_Server_Error, "Failed to open file");
                            return;
                        }

                        int num_row_groups = arrowReader->num_row_groups();
                        if (num_row_groups == 0) {
                            response.send(Http::Code::Internal_Server_Error, "No row groups found in the Parquet file.");
                            return;
                        }

                        auto filter_expression = GetFilterExpression(sqlExpression);

                        // Get select columns
                        auto select_columns = GetSelectColumns(sqlExpression);
                        bool is_aggregation = std::any_of(select_columns.begin(), select_columns.end(), [](const std::string& col) {
                            return col == "COUNT" || col == "SUM";
                        });

                        nlohmann::json aggregation_result;
                        std::shared_ptr<arrow::Table> final_table;

                        auto response_ptr = std::make_shared<Http::ResponseWriter>(std::move(response));

                        std::vector<std::future<void>> futures;
                        for (int row_group_index = 0; row_group_index < num_row_groups; ++row_group_index) {
                            futures.push_back(thread_pool.enqueue([=, response_ptr, bufferReader, &final_table, &aggregation_result] {
                                filterDataSync(filter_expression, bufferReader, row_group_index, response_ptr, aggregation_result, final_table, select_columns, is_aggregation);
                            }));
                        }

                        // Wait for all threads to complete
                        for (auto& future : futures) {
                            future.get();
                        }

                        auto end_filter = std::chrono::high_resolution_clock::now();
                        filterTime = end_filter - start_filter;
                        std::cout << "Time to filter: " << filterTime.count() << " seconds" << std::endl;

                        if (is_aggregation) {
                            response_ptr->send(Http::Code::Ok, aggregation_result.dump(), MIME(Application, Json));
                        } else {

                            auto start_write = std::chrono::high_resolution_clock::now();

                            if (output == "json") {
                                
                                std::string filtered_result_json = final_table->ToString();

                                response.send(Http::Code::Ok, filtered_result_json, MIME(Application, Json));

                            }else{
            
                               
                                /*std::shared_ptr<arrow::io::BufferOutputStream> buffer_output;
                                PARQUET_ASSIGN_OR_THROW(buffer_output, arrow::io::BufferOutputStream::Create());

                                PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
                                    *final_table,
                                    arrow::default_memory_pool(),
                                    buffer_output
                                ));

                                std::shared_ptr<arrow::Buffer> buffer;
                                PARQUET_ASSIGN_OR_THROW(buffer, buffer_output->Finish());*/

                                auto writer_properties = parquet::WriterProperties::Builder()
                                    .compression(parquet::Compression::SNAPPY)  // Correct usage with '.'
                                    ->build();                                 // Correct usage with '->'
                                std::shared_ptr<arrow::io::BufferOutputStream> buffer_output;
                                PARQUET_ASSIGN_OR_THROW(buffer_output, arrow::io::BufferOutputStream::Create());

                                PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
                                    *final_table,
                                    arrow::default_memory_pool(),
                                    buffer_output,
                                    1024 * 1024,  // Optional: size of row groups
                                    writer_properties                // Pass the writer properties with compression
                                ));

                                std::shared_ptr<arrow::Buffer> buffer;
                                PARQUET_ASSIGN_OR_THROW(buffer, buffer_output->Finish());


                                // Send non-encrypted results
                                response_ptr->send(Http::Code::Ok, buffer->ToString(), MIME(Application, OctetStream));
                            }

                            auto end_write = std::chrono::high_resolution_clock::now();
                            writeTime = end_write - start_write;
                            std::cout << "Write time taken: " << writeTime.count() << " seconds" << std::endl;
                        }

                        auto end_f = std::chrono::high_resolution_clock::now();
                        totalTime = end_f - start_f;
                        std::cout << "Total time taken: " << totalTime.count() << " seconds." << std::endl;

                        if (isFirstQuery) {
                            outputFile.open("../timing_results_web.txt");
                            if (outputFile.is_open()) {
                                outputFile << "Query #\t" << queryIndex << "\n";
                                outputFile << "Fetch\t" << std::fixed << std::setprecision(3) << fetchTime.count() << "\n";
                                outputFile << "Decrypt\t" << std::fixed << std::setprecision(3) << decryptTime.count() << "\n";
                                outputFile << "Filter\t" << std::fixed << std::setprecision(3) << filterTime.count() << "\n";
                                outputFile << "Write\t" << std::fixed << std::setprecision(3) << writeTime.count() << "\n";
                                outputFile << "Total\t" << std::fixed << std::setprecision(3) << totalTime.count() << "\n";
                                outputFile.close();
                                isFirstQuery = false;
                            } else {
                                std::cerr << "Unable to open file for writing." << std::endl;
                            }
                        } else {
                            std::ifstream inputFile("../timing_results_web.txt");
                            std::string line;
                            std::vector<std::string> lines;

                            if (inputFile.is_open()) {
                                while (std::getline(inputFile, line)) {
                                    lines.push_back(line);
                                }
                                inputFile.close();

                                outputFile.open("../timing_results_web.txt");
                                if (outputFile.is_open()) {
                                    int lineIndex = 0;
                                    outputFile << lines[lineIndex++] << "\t" << queryIndex << "\n";

                                    outputFile << lines[lineIndex++] << "\t" << std::fixed << std::setprecision(3) << fetchTime.count() << "\n";
                                    outputFile << lines[lineIndex++] << "\t" << std::fixed << std::setprecision(3) << decryptTime.count() << "\n";
                                    outputFile << lines[lineIndex++] << "\t" << std::fixed << std::setprecision(3) << filterTime.count() << "\n";
                                    outputFile << lines[lineIndex++] << "\t" << std::fixed << std::setprecision(3) << writeTime.count() << "\n";
                                    outputFile << lines[lineIndex++] << "\t" << std::fixed << std::setprecision(3) << totalTime.count() << "\n";
                                    outputFile.close();
                                } else {
                                    std::cerr << "Unable to open file for appending." << std::endl;
                                }
                            } else {
                                std::cerr << "Unable to open file for reading." << std::endl;
                            }
                        }

                        queryIndex++; // Increment query index for the next iteration



                    } else {
                        std::cerr << "Failed to get object parts" << std::endl;
                        response.send(Http::Code::Internal_Server_Error, "Failed to get object parts from S3");
                    }
                } else {
                    std::cerr << "Failed to get object metadata: " << headObjectOutcome.GetError().GetMessage() << std::endl;
                    response.send(Http::Code::Internal_Server_Error, "Failed to get object metadata from S3");
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception: " << e.what() << std::endl;
                response.send(Http::Code::Internal_Server_Error, "Exception occurred");
            }

            Aws::ShutdownAPI(options);
        } else {
            response.send(Http::Code::Not_Found, "Endpoint not found");
        }
    }

private:
    std::vector<char>& data;
    uint8_t* dst_buffer;
    uint8_t* final_buffer;
    std::vector<char>& decrypt_buffer;
};


int main() {

    if (setenv("AWS_EC2_METADATA_DISABLED", "true", 1) != 0) {}
       

    curl_global_init(CURL_GLOBAL_DEFAULT);

    init_crypto_resources();
    enc_init_crypto_resources();

    std::vector<char> data_buffer(alloc_size);
    std::vector<char> decrypt_buffer(max_rowgroup_size);

    uint8_t* dst_buf = prep_doca_buffer_dst(alloc_size);
    if (!dst_buf) {
        std::cerr << "Failed to prepare DOCA destination buffer." << std::endl;
        return 1;
    }

    prep_doca_buffer_src(alloc_size, data_buffer.data());

    uint8_t* final_buf = enc_prep_doca_buffer_dst(max_rowgroup_size);
    if (!dst_buf) {
        std::cerr << "Failed to prepare DOCA destination buffer." << std::endl;
        return 1;
    }

    enc_prep_doca_buffer_src(max_rowgroup_size, decrypt_buffer.data());

    Address addr(Ipv4::any(), Port(8080));
    auto opts = Http::Endpoint::options();
    auto endpoint = std::make_shared<Http::Endpoint>(addr);
    auto handler = std::make_shared<FilterHandler>(data_buffer, dst_buf, final_buf, decrypt_buffer);

    endpoint->init(opts);
    endpoint->setHandler(handler);
    endpoint->serve();

    stop_mmap();
    enc_stop_mmap();

    destroy_crypto_resources();
    enc_destroy_crypto_resources();

    curl_global_cleanup();

    return 0;
}
