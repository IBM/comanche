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

using namespace Pistache;
using json = nlohmann::json;

std::mutex decryption_mutex;
std::mutex response_mutex;

extern "C" {
    int encrypt_buffer(char* data, size_t size);
    uint8_t* decrypt_buffer(char* file_data, size_t file_size, size_t* output_size, uint8_t* dst_buffer);
    void init_crypto_resources();
    void destroy_crypto_resources();
    uint8_t* prep_doca_buffer(size_t file_size);
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
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 8 * 1024 * 1024); // Increased buffer size
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
    const size_t chunk_size = 8 * 1024 * 1024; // 8 MB
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
        auto start_prefetch = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < data_buffer.size(); i += 4096) {
            volatile char tmp = data_buffer[i];
        }
        auto end_prefetch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> prefetch_elapsed = end_prefetch - start_prefetch;
        std::cout << "Time taken to prefetch data buffer: " << prefetch_elapsed.count() << " seconds." << std::endl;


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

void PrepareBufferAsync(size_t file_size, std::promise<uint8_t*>&& promise) {
    thread_pool.enqueue([file_size, promise = std::move(promise)]() mutable {
        uint8_t* dst_buf = prep_doca_buffer(file_size);
        if (dst_buf) {
            promise.set_value(dst_buf);
        } else {
            std::cerr << "Failed to prepare DOCA buffer." << std::endl;
            promise.set_value(nullptr);
        }
    });
}

void decryptAndProcessData(char* data, size_t size, std::promise<uint8_t*>&& promise, size_t* output_size, uint8_t* dst_buffer) {
    thread_pool.enqueue([data, size, promise = std::move(promise), output_size, dst_buffer]() mutable {
        std::lock_guard<std::mutex> lock(decryption_mutex);
        uint8_t* decrypted_data = decrypt_buffer(data, size, output_size, dst_buffer);
        if (decrypted_data) {
            std::cerr << "Decryption successful. Output size: " << *output_size << std::endl;
            promise.set_value(decrypted_data);
        } else {
            std::cerr << "Decryption failed" << std::endl;
            promise.set_value(nullptr);
        }
    });
}

void stopMmapAsync() {
    thread_pool.enqueue([]() {
        stop_mmap();
    });
}

arrow::compute::Expression GetFilterExpression(const std::string& sqlExpression, std::string& columnName, int& l_value) {
    std::vector<Token> tokens = SQLParser::parse(sqlExpression);

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
    }

    auto field_ref = arrow::compute::field_ref(columnName);
    l_value = std::stoi(value);
    auto literal_value = arrow::compute::literal(l_value);

    if (operatorSymbol == "=") {
        return arrow::compute::equal(field_ref, literal_value);
    } else if (operatorSymbol == ">") {
        return arrow::compute::greater(field_ref, literal_value);
    } else if (operatorSymbol == ">=") {
        return arrow::compute::greater_equal(field_ref, literal_value);
    } else if (operatorSymbol == "<") {
        return arrow::compute::less(field_ref, literal_value);
    } else if (operatorSymbol == "<=") {
        return arrow::compute::less_equal(field_ref, literal_value);
    } else if (operatorSymbol == "!=") {
        return arrow::compute::not_equal(field_ref, literal_value);
    }

    throw std::invalid_argument("Unsupported operator in SQL expression");
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

void filterDataAsync(const arrow::compute::Expression& filter_expression, std::shared_ptr<arrow::io::BufferReader> bufferReader, int row_group_index, std::shared_ptr<Http::ResponseWriter> response) {
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

        std::string filtered_result_json = result_table->ToString();

        std::lock_guard<std::mutex> lock(response_mutex);
        response->send(Http::Code::Ok, filtered_result_json, MIME(Application, Json));
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

struct FilterHandler : public Http::Handler {
    HTTP_PROTOTYPE(FilterHandler)

    FilterHandler(std::vector<std::vector<char>>& buffers, std::vector<char>& data_buf)
        : chunk_buffers(buffers), data_buffer(data_buf) {}

    void onRequest(const Http::Request& req, Http::ResponseWriter response) override {
        if (req.resource() == "/data" && req.method() == Http::Method::Post) {
            json requestJson = json::parse(req.body());

            std::string bucket = requestJson["bucket"];
            std::string key = requestJson["key"];
            std::string sqlExpression = requestJson["sql"];

            auto start_f = std::chrono::high_resolution_clock::now();

            std::string url = "http://10.10.10.18/encrypted_600000.parquet"; // Change to your actual URL

            size_t content_size = 0;

            try {
                if (GetContentSize(url, content_size)) {
                    // Create a promise for buffer preparation
                    std::promise<uint8_t*> buffer_promise;
                    std::future<uint8_t*> buffer_future = buffer_promise.get_future();

                    auto start_alloc = std::chrono::high_resolution_clock::now();

                    // Start asynchronous buffer preparation
                    PrepareBufferAsync(content_size, std::move(buffer_promise));

                    std::cout << "Content size: " << content_size << " bytes." << std::endl;

                    // Resize data buffer to content size
                    
                    data_buffer.resize(content_size);

                    auto end_alloc = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> alloc_elapsed = end_alloc - start_alloc;
                    std::cout << "Time taken to allocate data buffer: " << alloc_elapsed.count() << " seconds." << std::endl;

                    auto start_time = std::chrono::high_resolution_clock::now();

                

                    if (DownloadFileParallel(url, content_size, data_buffer, chunk_buffers)) {
                        auto end_time = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> elapsed = end_time - start_time;
                        std::cout << "Data fetched successfully. Total time taken to fetch data: " << elapsed.count() << " seconds." << std::endl;

                        auto start_decrypt = std::chrono::high_resolution_clock::now();

                        uint8_t* dst_buf = buffer_future.get();

                        size_t output_size = 0; // Declare output_size here

                        
                        // Create a promise for decryption
                        std::promise<uint8_t*> decryption_promise;
                        std::future<uint8_t*> decryption_future = decryption_promise.get_future();

                        // Start decryption asynchronously
                        decryptAndProcessData(data_buffer.data(), data_buffer.size(), std::move(decryption_promise), &output_size, dst_buf);

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
                            auto filter_expression = GetFilterExpression(sqlExpression, columnName, l_value);

                            auto arrowBuffer = arrow::Buffer::Wrap(decrypted_data, output_size);
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

                            auto matching_row_groups = GetMatchingRowGroups(arrowReader, columnName, l_value);

                            // Create a shared pointer for the response writer
                            auto shared_response = std::make_shared<Http::ResponseWriter>(std::move(response));

                            if (!matching_row_groups.empty()) {
                                for (const auto& row_group_index : matching_row_groups) {
                                    thread_pool.enqueue([=, shared_response, bufferReader] {
                                        filterDataAsync(filter_expression, bufferReader, row_group_index, shared_response);
                                    });
                                }
                            } else {
                                for (int row_group_index = 0; row_group_index < num_row_groups; ++row_group_index) {
                                    thread_pool.enqueue([=, shared_response, bufferReader] {
                                        filterDataAsync(filter_expression, bufferReader, row_group_index, shared_response);
                                    });
                                }
                            }

                            auto end_filter = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> filter_elapsed = end_filter - start_filter;
                            std::cout << "Filtering time taken: " << filter_elapsed.count() << " seconds." << std::endl;

                            auto end_f = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> total_elapsed = end_f - start_f;
                            std::cout << "Total time taken: " << total_elapsed.count() << " seconds." << std::endl;

                            stopMmapAsync();

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
};

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    init_crypto_resources();

    const size_t chunk_size = 8 * 1024 * 1024; // 8 MB
    size_t num_chunks = 1024 / 8;
    std::vector<std::vector<char>> chunk_buffers(num_chunks, std::vector<char>(chunk_size)); // Pre-allocate chunk buffers

    // Pre-allocate data buffer with a size of 1GB
    std::vector<char> data_buffer(1 * 1024 * 1024 * 1024); // 1GB

    Address addr(Ipv4::any(), Port(8080));
    auto opts = Http::Endpoint::options();
    auto endpoint = std::make_shared<Http::Endpoint>(addr);
    auto handler = std::make_shared<FilterHandler>(chunk_buffers, data_buffer);

    endpoint->init(opts);
    endpoint->setHandler(handler);
    endpoint->serve();

    destroy_crypto_resources();

    curl_global_cleanup();

    return 0;
}