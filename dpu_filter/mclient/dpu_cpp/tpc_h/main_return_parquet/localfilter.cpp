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
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/exec.h>
#include <ctime>
#include <arrow/compute/expression.h>
#include "/home/ubuntu/json/include/nlohmann/json.hpp"
#include "SQLParser.h"
#include <future>
#include <thread>
#include <mutex>
#include <atomic>
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


using namespace Pistache;
using json = nlohmann::json;

std::mutex decryption_mutex;
std::mutex response_mutex;
std::mutex table_mutex;

const size_t chunk_size = 8 * 1024 * 1024; // 8 MB
const size_t alloc_size = 1024 * 1024 * 1024; // 1 GB for memory allocations

extern "C" {
    //Decrypt functions
    uint8_t* decrypt_buffer(char* file_data, size_t file_size, size_t* output_size, uint8_t* dst_buffer);
    void init_crypto_resources();
    void destroy_crypto_resources();
    uint8_t* prep_doca_buffer_dst(const size_t file_size);
    uint8_t* prep_doca_buffer_src(const size_t file_size, char* file_data);
    void stop_mmap();
}

// Thread pool class
class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    void set_affinity(std::thread::native_handle_type handle, int core_id);
};

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
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

struct Chunk {
    size_t start;
    size_t end;
    std::vector<char>* data_buffer;
    size_t received_size; // track the amount of data received
};

// Function to concatenate tables
std::shared_ptr<arrow::Table> ConcatenateTables(const std::shared_ptr<arrow::Table>& table1, const std::shared_ptr<arrow::Table>& table2) {
    std::vector<std::shared_ptr<arrow::Table>> tables = {table1, table2};
    auto result = arrow::ConcatenateTables(tables);
    if (!result.ok()) {
        throw std::runtime_error("Failed to concatenate tables");
    }
    return *result;
}


size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    Chunk* chunk = static_cast<Chunk*>(userp);

    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, chunk->data_buffer->begin() + chunk->received_size);
    chunk->received_size += totalSize; // update the received size
    return totalSize;
}

bool DownloadChunk(const std::string& url, Chunk& chunk) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl initialization failed" << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, chunk_size); // Increased buffer size
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &chunk);
    curl_easy_setopt(curl, CURLOPT_RANGE, (std::to_string(chunk.start) + "-" + std::to_string(chunk.end)).c_str());
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
    curl_easy_setopt(curl, CURLOPT_MAX_RECV_SPEED_LARGE, (curl_off_t)0);
    curl_easy_setopt(curl, CURLOPT_MAX_SEND_SPEED_LARGE, (curl_off_t)0);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_cleanup(curl);
    return true;
}

bool GetContentSize(const std::string& url, size_t& content_size) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "CURL initialization failed." << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_HEADER, 1L);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, +[](void* buffer, size_t size, size_t nitems, void* userdata) -> size_t {
        std::string header((char*)buffer, size * nitems);
        std::string content_length_key = "Content-Length: ";
        auto found = header.find(content_length_key);
        if (found != std::string::npos) {
            size_t content_length = std::stoull(header.substr(found + content_length_key.size()));
            *static_cast<size_t*>(userdata) = content_length;
        }
        return nitems * size;
    });
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &content_size);

    // Enable HTTP/2
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }

    return true;
}

bool DownloadFileParallel(const std::string& url, size_t content_size, std::vector<char>& data_buffer, std::vector<std::vector<char>>& chunk_buffers) {
    
    size_t num_chunks = (content_size + chunk_size - 1) / chunk_size;

    std::vector<Chunk> chunks(num_chunks);

    for (size_t i = 0; i < num_chunks; ++i) {
        chunks[i].start = i * chunk_size;
        chunks[i].end = std::min((i + 1) * chunk_size, content_size) - 1;
        chunks[i].data_buffer = &chunk_buffers[i]; // Assign pre-allocated buffer
        chunks[i].received_size = 0; // Initialize received size
    }

    std::vector<std::future<bool>> download_futures;
    auto start_download = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_chunks; ++i) {
        download_futures.push_back(thread_pool.enqueue([&, i] {
            return DownloadChunk(url, chunks[i]);
        }));
    }

    bool success = true;
    for (size_t i = 0; i < download_futures.size(); ++i) {
        if (!download_futures[i].get()) {
            success = false;
        }
    }

    auto end_download = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> download_elapsed = end_download - start_download;
    std::cout << "Time taken to download all chunks: " << download_elapsed.count() << " seconds." << std::endl;

    if (success) {

        // Prefetch the data buffer before aggregation
        //auto start_prefetch = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < data_buffer.size(); i += 4096) {
            volatile char tmp = data_buffer[i];
        }
        /*auto end_prefetch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> prefetch_elapsed = end_prefetch - start_prefetch;
        std::cout << "Time taken to prefetch data buffer: " << prefetch_elapsed.count() << " seconds." << std::endl;*/


        auto start_aggregation = std::chrono::high_resolution_clock::now();
        std::vector<std::future<void>> aggregation_futures;
        for (const auto& chunk : chunks) {
            aggregation_futures.push_back(thread_pool.enqueue([&data_buffer, &chunk] {
                std::memcpy(data_buffer.data() + chunk.start, chunk.data_buffer->data(), chunk.received_size);
            }));
        }

        for (auto& future : aggregation_futures) {
            future.get();
        }

        auto end_aggregation = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> aggregation_elapsed = end_aggregation - start_aggregation;
        std::cout << "Time taken to aggregate all chunks: " << aggregation_elapsed.count() << " seconds." << std::endl;
    }

    return success;
}



void decryptAndProcessData(char* data, size_t size, std::promise<uint8_t*>&& promise, size_t* output_size, uint8_t* dst_buffer) {
    thread_pool.enqueue([data, size, promise = std::move(promise), output_size, dst_buffer]() mutable {
        std::lock_guard<std::mutex> lock(decryption_mutex);
        uint8_t* decrypted_data = decrypt_buffer(data, size, output_size, dst_buffer);
        if (decrypted_data) {
            //std::cerr << "Decryption successful. Output size: " << *output_size << std::endl;
            promise.set_value(decrypted_data);
        } else {
            std::cerr << "Decryption failed" << std::endl;
            promise.set_value(nullptr);
        }
    });
}




/*arrow::compute::Expression GetFilterExpression(const std::string& sqlExpression) {
    std::vector<Token> tokens = SQLParser::parse(sqlExpression);

    std::string columnName;
    std::string operatorSymbol;
    std::string value;

    // Debug: print tokens
    std::cout << "Parsed tokens:" << std::endl;
    for (const auto& token : tokens) {
        std::cout << "Type: " << SQLParser::tokenTypeToString(token.type) << ", Value: " << token.value << std::endl;
        if (token.type == TokenType::COLUMN) {
            columnName = token.value;
        } else if (token.type == TokenType::OPERATOR) {
            operatorSymbol = token.value;
        } else if (token.type == TokenType::LITERAL) {
            value = token.value;
        }
    }

    // Debug: print extracted components
    std::cout << "Column: " << columnName << ", Operator: " << operatorSymbol << ", Value: " << value << std::endl;

    if (columnName.empty() || operatorSymbol.empty() || value.empty()) {
        throw std::invalid_argument("Invalid SQL expression");
    }

    // Remove quotes from the value if present
    if (value.front() == '\'' && value.back() == '\'') {
        value = value.substr(1, value.size() - 2);
    }

    auto field_ref = arrow::compute::field_ref(columnName);

    // Handle different data types
    arrow::compute::Expression condition;
    if (columnName == "l_shipdate") {
        // Parse the date string into a timestamp
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
        } else {
            throw std::invalid_argument("Unsupported operator in SQL expression");
        }
    } else if (columnName == "l_quantity" || columnName == "l_extendedprice" || columnName == "l_discount") {
        // Handle double type
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
        } else {
            throw std::invalid_argument("Unsupported operator in SQL expression");
        }
    } else if (columnName == "l_orderkey" || columnName == "l_partkey" || columnName == "l_suppkey" || columnName == "l_linenumber" || columnName == "__index_level_0__") {
        // Handle int64 type
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
        } else {
            throw std::invalid_argument("Unsupported operator in SQL expression");
        }
    } else {
        // Handle string type
        auto literal_value = arrow::compute::literal(value);

        if (operatorSymbol == "=") {
            condition = arrow::compute::equal(field_ref, literal_value);
        } else if (operatorSymbol == "!=") {
            condition = arrow::compute::not_equal(field_ref, literal_value);
        } else {
            throw std::invalid_argument("Unsupported operator in SQL expression");
        }
    }

    return condition;
}*/



arrow::compute::Expression GetFilterExpression(const std::string& sqlExpression) {
    std::vector<Token> tokens = SQLParser::parse(sqlExpression);

    std::vector<arrow::compute::Expression> conditions;
    std::string columnName;
    std::string operatorSymbol;
    std::string value;
    std::string value2;

    // Debug: print tokens
    std::cout << "Parsed tokens:" << std::endl;
    for (const auto& token : tokens) {
        std::cout << "Type: " << SQLParser::tokenTypeToString(token.type) << ", Value: " << token.value << std::endl;
    }

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
                        // Handle int64 and string types
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

                    // Debug: print the constructed condition
                    std::cout << "Constructed condition for column " << columnName << ": " << condition.ToString() << std::endl;

                    conditions.push_back(condition);
                }
            }
        }

        // Combine all conditions using 'and'
        if (!conditions.empty()) {
            arrow::compute::Expression combined_condition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                std::cout << "Combining condition: " << combined_condition.ToString() << " with " << conditions[i].ToString() << std::endl;
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




/*void filterDataAsync(const arrow::compute::Expression& filter_expression, std::shared_ptr<arrow::io::BufferReader> bufferReader, int row_group_index, std::shared_ptr<Http::ResponseWriter> response, std::shared_ptr<arrow::Table>& final_table) {
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

        // Print the number of rows in the filtered table for each row group
       // std::cout << "Number of rows in the filtered table for row group " << row_group_index << ": " << result_table->num_rows() << std::endl;

        // Merge individual tables into the final table
        std::lock_guard<std::mutex> lock(table_mutex);
        if (final_table == nullptr) {
            final_table = result_table;
        } else {
            final_table = ConcatenateTables(final_table, result_table);
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(response_mutex);
        response->send(Http::Code::Internal_Server_Error, e.what());
    }
}*/





void filterDataAsync(const arrow::compute::Expression& filter_expression, std::shared_ptr<arrow::io::BufferReader> bufferReader, int row_group_index, std::shared_ptr<Http::ResponseWriter> response, std::shared_ptr<arrow::Table>& final_table) {
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

        // Compute the revenue
        auto extendedprice = result_table->GetColumnByName("l_extendedprice");
        auto discount = result_table->GetColumnByName("l_discount");

        auto multiply_result = arrow::compute::CallFunction("multiply", {extendedprice->chunk(0), discount->chunk(0)});
        if (!multiply_result.ok()) {
            std::lock_guard<std::mutex> lock(response_mutex);
            response->send(Http::Code::Internal_Server_Error, "Failed to multiply columns.");
            return;
        }
        auto revenue_column = multiply_result.ValueOrDie().make_array();

        auto sum_result = arrow::compute::Sum(revenue_column);
        if (!sum_result.ok()) {
            std::lock_guard<std::mutex> lock(response_mutex);
            response->send(Http::Code::Internal_Server_Error, "Failed to compute sum.");
            return;
        }
        auto revenue = std::static_pointer_cast<arrow::DoubleScalar>(sum_result.ValueOrDie().scalar())->value;

        // Create the result table with the revenue column
        auto revenue_array = std::make_shared<arrow::DoubleArray>(1, arrow::Buffer::Wrap(std::vector<double>{revenue}));
        auto schema = arrow::schema({arrow::field("revenue", arrow::float64())});
        auto result_table_with_revenue = arrow::Table::Make(schema, {revenue_array});

        // Merge individual tables into the final table
        std::lock_guard<std::mutex> lock(table_mutex);
        if (final_table == nullptr) {
            final_table = result_table_with_revenue;
        } else {
            std::shared_ptr<arrow::Table> combined_table;
            std::vector<std::shared_ptr<arrow::Table>> tables_to_concatenate = {final_table, result_table_with_revenue};
            auto concatenate_result = arrow::ConcatenateTables(tables_to_concatenate, arrow::ConcatenateTablesOptions::Defaults());
            if (!concatenate_result.ok()) {
                std::lock_guard<std::mutex> lock(response_mutex);
                response->send(Http::Code::Internal_Server_Error, "Failed to concatenate tables.");
                return;
            }
            combined_table = concatenate_result.ValueOrDie();
            final_table = combined_table;
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

class FilterHandler : public Http::Handler {
    HTTP_PROTOTYPE(FilterHandler)

    FilterHandler(std::vector<std::vector<char>>& buffers, std::vector<char>& data_buf, uint8_t* dst_buf)
        : chunk_buffers(buffers), data_buffer(data_buf), dst_buffer(dst_buf) {}

    void onRequest(const Http::Request& req, Http::ResponseWriter response) override {
        if (req.resource() == "/data" && req.method() == Http::Method::Post) {
            json requestJson = json::parse(req.body());

            std::string bucket = requestJson["bucket"];
            std::string key = requestJson["key"];
            std::string sqlExpression = requestJson["sql"];

            auto start_f = std::chrono::high_resolution_clock::now();

            std::string url = "http://10.10.10.18/parquet_files/"+key; // Change to your actual URL

            size_t content_size = 0;

            std::shared_ptr<arrow::Table> final_table;


            try {
                if (GetContentSize(url, content_size)) {
                    std::cout << "Content size: " << content_size << " bytes." << std::endl;

                    // Resize data buffer to content size
                    data_buffer.resize(content_size);

                    auto start_time = std::chrono::high_resolution_clock::now();

                    if (DownloadFileParallel(url, content_size, data_buffer, chunk_buffers)) {
                        auto end_time = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> elapsed = end_time - start_time;
                        std::cout << "Data fetched successfully. Total time taken to fetch data: " << elapsed.count() << " seconds." << std::endl;

                        auto start_decrypt = std::chrono::high_resolution_clock::now();

                        size_t output_size = 0; // Declare output_size here

                        // Create a promise for decryption
                        std::promise<uint8_t*> decryption_promise;
                        std::future<uint8_t*> decryption_future = decryption_promise.get_future();

                        // Start decryption asynchronously
                        decryptAndProcessData(data_buffer.data(), data_buffer.size(), std::move(decryption_promise), &output_size, dst_buffer);

                        // Wait for decryption to complete
                        uint8_t* decrypted_data = decryption_future.get();

                        auto end_decrypt = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> decrypt_elapsed = end_decrypt - start_decrypt;
                        std::cout << "Decryption time taken: " << decrypt_elapsed.count() << " seconds." << std::endl;

                        if (decrypted_data) {
                            std::cerr << "Decryption successful. Output size: " << output_size << std::endl;

                            auto start_filter = std::chrono::high_resolution_clock::now();

                            // Extract column name and literal value
                            std::string columnName;
                            int l_value;
                            // Create the filter expression
                          //  


                            auto arrowBuffer = arrow::Buffer::Wrap(decrypted_data, output_size);
                            auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);
                            std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                            auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);
                            if (!status.ok()) {
                                response.send(Http::Code::Internal_Server_Error, "Failed to open file");
                                return;
                            }



 std::shared_ptr<arrow::Schema> schema;
    PARQUET_THROW_NOT_OK(arrowReader->GetSchema(&schema));

    // Print the schema
    std::cout << "Schema of the Parquet file:" << std::endl;
    std::cout << schema->ToString() << std::endl;


auto filter_expression = GetFilterExpression(sqlExpression);

std::cout << "Got expression" << std::endl;

                            int num_row_groups = arrowReader->num_row_groups();
                            if (num_row_groups == 0) {
                                response.send(Http::Code::Internal_Server_Error, "No row groups found in the Parquet file.");
                                return;
                            }

                           //auto matching_row_groups = GetMatchingRowGroups(arrowReader, columnName, l_value);

                            // Create a shared pointer for the response writer
                            auto shared_response = std::make_shared<Http::ResponseWriter>(std::move(response));

                  


                                              // Process each row group in parallel
                        std::vector<std::future<void>> futures;
                        for (int row_group_index = 0; row_group_index < num_row_groups; ++row_group_index) {
                            futures.push_back(thread_pool.enqueue([=, shared_response, bufferReader, &final_table] {
                                filterDataAsync(filter_expression, bufferReader, row_group_index, shared_response, final_table);
                            }));
                        }

                        // Wait for all threads to complete
                        for (auto& future : futures) {
                            future.get();
                        }

                            auto end_filter = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> filter_elapsed = end_filter - start_filter;
                            std::cout << "Filtering time taken: " << filter_elapsed.count() << " seconds." << std::endl;

                            auto end_f = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> total_elapsed = end_f - start_f;
                            std::cout << "Total time taken: " << total_elapsed.count() << " seconds." << std::endl;

                                // Convert the final table to Parquet and send as response
                        std::shared_ptr<arrow::io::BufferOutputStream> buffer_output;
                        PARQUET_ASSIGN_OR_THROW(buffer_output, arrow::io::BufferOutputStream::Create());

                        PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
                            *final_table,
                            arrow::default_memory_pool(),
                            buffer_output,
                            1024 * 1024 // 1MB row group size
                        ));

                        std::shared_ptr<arrow::Buffer> buffer;
                        PARQUET_ASSIGN_OR_THROW(buffer, buffer_output->Finish());

                        std::lock_guard<std::mutex> lock(response_mutex);
                        shared_response->send(Http::Code::Ok, buffer->ToString(), MIME(Application, OctetStream));
              

                        } else {
                            std::cout << "Decryption failed." << std::endl;
                            response.send(Http::Code::Internal_Server_Error, "Decryption failed");
                        }

                    } else {
                        std::cerr << "Data fetch failed" << std::endl;
                        response.send(Http::Code::Internal_Server_Error, "Data fetch failed");
                    }
                } else {
                    std::cerr << "Failed to retrieve content size." << std::endl;
                    response.send(Http::Code::Internal_Server_Error, "Failed to retrieve content size");
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception: " << e.what() << std::endl;
                response.send(Http::Code::Internal_Server_Error, "Exception occurred");
            }
        } else {
            response.send(Http::Code::Not_Found, "Endpoint not found");
        }
    }

private:
    std::vector<std::vector<char>>& chunk_buffers;
    std::vector<char>& data_buffer;
    uint8_t* dst_buffer;
};


int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    init_crypto_resources();


    size_t num_chunks = alloc_size/ chunk_size;
    std::vector<std::vector<char>> chunk_buffers(num_chunks, std::vector<char>(chunk_size)); // Pre-allocate chunk buffers

    // Pre-allocate data buffer with a size of 1GB
    std::vector<char> data_buffer(alloc_size); // 1GB

    // Prepare destination buffer with 1GB size synchronously
    uint8_t* dst_buf = prep_doca_buffer_dst(alloc_size);
    if (!dst_buf) {
        std::cerr << "Failed to prepare DOCA destination buffer." << std::endl;
        return 1;
    }

    // Prepare source buffer with 1GB size synchronously
    prep_doca_buffer_src(alloc_size, data_buffer.data());

    Address addr(Ipv4::any(), Port(8080));
    auto opts = Http::Endpoint::options();
    auto endpoint = std::make_shared<Http::Endpoint>(addr);
    auto handler = std::make_shared<FilterHandler>(chunk_buffers, data_buffer, dst_buf);

    endpoint->init(opts);
    endpoint->setHandler(handler);
    endpoint->serve();

    stop_mmap();

    destroy_crypto_resources();

    curl_global_cleanup();

    return 0;
}
