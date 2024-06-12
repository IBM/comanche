#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <locale>
#include <sys/mman.h>
#include <pistache/endpoint.h>
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

class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> task_queue;
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
                    this->condition.wait(lock, [this] { return this->stop || !this->task_queue.empty(); });
                    if (this->stop && this->task_queue.empty())
                        return;
                    task = std::move(this->task_queue.front());
                    this->task_queue.pop();
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
        task_queue.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

ThreadPool download_thread_pool(std::thread::hardware_concurrency());
ThreadPool aggregation_thread_pool(std::thread::hardware_concurrency());

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
    const size_t chunk_size = 8 * 1024 * 1024; // 16 MB
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
        download_futures.push_back(download_thread_pool.enqueue([&, i] {
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
        auto start_aggregation = std::chrono::high_resolution_clock::now();
        std::vector<std::future<void>> aggregation_futures;
        for (const auto& chunk : chunks) {
            aggregation_futures.push_back(aggregation_thread_pool.enqueue([&data_buffer, &chunk] {
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

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::string url = "http://10.10.10.18/dataStat_600000.parquet"; // Change to your actual URL
    size_t content_size = 0;

    if (GetContentSize(url, content_size)) {
        std::cout << "Content size: " << content_size << " bytes." << std::endl;
        std::vector<char> data_buffer(content_size);

        const size_t chunk_size = 8 * 1024 * 1024; // 16 MB
        size_t num_chunks = 1024 / 8;
        //Pre allocating 1GB buffer chunks as that is max file size
        std::vector<std::vector<char>> chunk_buffers(num_chunks, std::vector<char>(chunk_size)); // Pre-allocate chunk buffers

        auto start_time = std::chrono::high_resolution_clock::now();


        if (DownloadFileParallel(url, content_size, data_buffer, chunk_buffers)) {
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            std::cout << "Data fetched successfully. Total time taken to fetch data: " << elapsed.count() << " seconds." << std::endl;


            auto arrowBuffer = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(data_buffer.data()), data_buffer.size());
            std::cout << "Arrow buffer created successfully." << std::endl;

            auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);
            std::cout << "BufferReader created successfully." << std::endl;

            std::unique_ptr<parquet::arrow::FileReader> arrowReader;
            auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);
            if (!status.ok()) {
                std::cerr << "Failed to open parquet file: " << status.ToString() << std::endl;
                return 1;
            }
            std::cout << "Parquet file opened successfully." << std::endl;

            std::shared_ptr<arrow::Table> table;
            status = arrowReader->ReadTable(&table);
            if (!status.ok()) {
                std::cerr << "Failed to read parquet table: " << status.ToString() << std::endl;
                return 1;
            }
            std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;

        } else {
            std::cerr << "Data fetch failed" << std::endl;
        }
    } else {
        std::cerr << "Failed to retrieve content size." << std::endl;
    }

    curl_global_cleanup();

    return 0;
}
