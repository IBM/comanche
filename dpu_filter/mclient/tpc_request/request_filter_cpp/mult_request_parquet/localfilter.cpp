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
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>

// Configuration constants
const std::string URL = "http://10.10.10.20:8080/data";
const std::string PAYLOAD = R"({
    "bucket": "encrypted",
    "key": "enc_inventory.parquet",
    "sql": "SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2452089"
})";
const std::string OUTPUT_FILENAME = "filtered_output.parquet";
const size_t CHUNK_SIZE = 8 * 1024 * 1024; // 8 MB
const size_t INITIAL_BUFFER_SIZE = 1024 * 1024 * 1024; // 1 GB

// Chunk structure
struct Chunk {
    size_t start;
    size_t end;
    std::vector<char>* data_buffer;
    size_t received_size; // track the amount of data received
};

// ThreadPool class
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

    // Resize the buffer if needed
    if (currentOffset + totalSize > memory.size()) {
        memory.resize(currentOffset + totalSize);
    }

    // Copy the received data into the vector at the current offset
    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, memory.begin() + currentOffset);
    currentOffset += totalSize; // Update the offset

    return totalSize;
}

bool SaveResponseToFile(const std::string& filename, const std::vector<char>& data, size_t dataSize) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return false;
    }
    outfile.write(data.data(), dataSize);
    if (!outfile) {
        std::cerr << "Error: Failed to write data to file: " << filename << std::endl;
        return false;
    }
    return true;
}

void PrintParquetFileContent(const std::string& filename) {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(filename));
    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

    // Print the total number of rows in the table
    std::cout << "Total number of rows in the filtered table: " << table->num_rows() << std::endl;
}

CURLcode PerformCurlRequest(const std::string& url, const std::string& payload, std::vector<char>& response_memory, struct curl_slist* headers) {
    // Initialize CURL
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return CURLE_FAILED_INIT;
    }

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

    // Clean up
    curl_easy_cleanup(curl);

    return res;
}

size_t WriteChunkCallback(void* contents, size_t size, size_t nmemb, void* userp) {
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
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteChunkCallback);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, CHUNK_SIZE); // Increased buffer size
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
    size_t num_chunks = (content_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    std::vector<Chunk> chunks(num_chunks);

    for (size_t i = 0; i < num_chunks; ++i) {
        chunks[i].start = i * CHUNK_SIZE;
        chunks[i].end = std::min((i + 1) * CHUNK_SIZE, content_size) - 1;
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
int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    // Headers
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Variables to store the response and timing
    std::vector<char> response_memory;
    response_memory.resize(INITIAL_BUFFER_SIZE); // Resize to initial buffer size
    ResetWriteMemoryCallbackOffset(); // Reset offset before starting

    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform the CURL request
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return 1;
    }

    curl_easy_setopt(curl, CURLOPT_URL, URL.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, PAYLOAD.c_str());
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
            std::cout << "Success: Parquet file saved as " << OUTPUT_FILENAME << std::endl;

            size_t content_size = 0;
            if (GetContentSize(URL, content_size)) {
                std::vector<std::vector<char>> chunk_buffers((content_size + CHUNK_SIZE - 1) / CHUNK_SIZE, std::vector<char>(CHUNK_SIZE));
                std::vector<char> data_buffer(content_size);

                if (DownloadFileParallel(URL, content_size, data_buffer, chunk_buffers)) {
                    if (SaveResponseToFile(OUTPUT_FILENAME, data_buffer, content_size)) {
                        PrintParquetFileContent(OUTPUT_FILENAME);
                    } else {
                        std::cerr << "Error saving response to file." << std::endl;
                    }
                } else {
                    std::cerr << "Error downloading file in parallel." << std::endl;
                }
            } else {
                std::cerr << "Error getting content size." << std::endl;
            }
        } else {
            std::cerr << "HTTP Error: " << http_code << std::endl;
        }
    }

    // Clean up
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    curl_global_cleanup();

    std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
