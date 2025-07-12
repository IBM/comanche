#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

using json = nlohmann::json;
std::mutex cout_mutex;

thread_local size_t currentOffset = 0;

void ResetWriteMemoryCallbackOffset() {
    currentOffset = 0;
}

size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto& memory = *static_cast<std::vector<char>*>(userp);
    size_t totalSize = size * nmemb;
    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, memory.begin() + currentOffset);
    currentOffset += totalSize;
    return totalSize;
}

std::string TrimQuotes(const std::string& str) {
    if (str.length() >= 2 && str.front() == '"' && str.back() == '"') {
        return str.substr(1, str.length() - 2);
    }
    return str;
}

std::vector<std::string> readQueriesFromFile(const std::string& filename) {
    std::vector<std::string> queries;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            queries.push_back(TrimQuotes(line));
        }
    }
    return queries;
}

void send_request(const std::string& url, const std::string& payload, int query_id) {
    CURL* curl = curl_easy_init();
    if (!curl) return;

    std::vector<char> response_memory(1024 * 1024 * 1024);
    ResetWriteMemoryCallbackOffset();

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_memory);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 2 * 1024 * 1024);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 120L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 60L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

    auto start = std::chrono::high_resolution_clock::now();
    CURLcode res = curl_easy_perform(curl);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    double bytes_received = 0;
    long num_rows = 0;
    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_SIZE_DOWNLOAD, &bytes_received);
        char* encoding = NULL;
        curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &encoding);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code == 200) {
            std::string output_filename = "filtered_output_" + std::to_string(query_id) + ".parquet";
            std::ofstream outfile(output_filename, std::ios::binary);
            outfile.write(response_memory.data(), currentOffset);
            outfile.close();
            std::shared_ptr<arrow::io::ReadableFile> infile;
            PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(output_filename));
            std::unique_ptr<parquet::arrow::FileReader> reader;
            PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
            std::shared_ptr<arrow::Table> table;
            PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
            num_rows = table->num_rows();
        }
    }

    //std::lock_guard<std::mutex> lock(cout_mutex);
   // std::cout << "Query " << query_id << " finished in " << elapsed << "s, Rows: " << num_rows << ", Bytes: " << bytes_received << std::endl;

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
}

int main() {
    curl_global_init(CURL_GLOBAL_ALL);
    std::string url = "http://10.10.10.20:8080/data";
    std::vector<std::string> queries = readQueriesFromFile("../queries.txt");
    std::string query = queries.front();

     std::vector<int> concurrency_levels(16);
     std::iota(concurrency_levels.begin(), concurrency_levels.end(), 1);

    int level = 2;


    //for (int level : concurrency_levels) {
        std::vector<std::thread> threads;
        auto start_all = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < level; ++i) {
            std::string file_id = std::to_string(i + 1);  // 1 to 9
            std::string key = "enc_inventory_" + file_id + ".parquet";
            //std::string key = "inventory_" + file_id + ".parquet";

            std::string payload = R"({"bucket":"encrypted","key":")" + key +
                                R"(","output":"parquet","sql":")" + query + R"("})";

            threads.emplace_back(send_request, url, payload, i + 1);
        }
        for (auto& t : threads) t.join();
        auto end_all = std::chrono::high_resolution_clock::now();
        double total_elapsed = std::chrono::duration<double>(end_all - start_all).count();

        std::cout << "[Concurrency: " << level << "] Total time: " << total_elapsed
                  << "s, Throughput: " << level / total_elapsed << " QPS" << std::endl << std::endl;
   //}

    curl_global_cleanup();
    return 0;
}
