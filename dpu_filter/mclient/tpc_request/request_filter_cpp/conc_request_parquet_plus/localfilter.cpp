#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <curl/curl.h>
#include <vector>
#include <nlohmann/json.hpp>
#include <iomanip>

using json = nlohmann::json;

static size_t& GetCurrentOffset() {
    static size_t currentOffset = 0;
    return currentOffset;
}

void ResetWriteMemoryCallbackOffset() {
    GetCurrentOffset() = 0;
}

size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto& memory = *static_cast<std::vector<char>*>(userp);
    size_t totalSize = size * nmemb;
    size_t& currentOffset = GetCurrentOffset();

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

int main() {
    std::string url = "http://10.10.10.20:8080/data";

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return 1;
    }

    std::vector<std::string> queries = readQueriesFromFile("../query10.txt");

    std::ofstream timingFile;
    bool isFirstQuery = true;
    int queryIndex = 1;

    for (const auto& sqlQuery : queries) {
        std::cout << "Processing query #" << queryIndex << ": " << sqlQuery << std::endl;

        std::string payload = R"({
            "bucket": "encrypted",
            "key": "enc_web_sales.parquet",
            "output": "parquet",
            "sql": ")" + sqlQuery + R"("
        })";

        std::vector<char> response_memory(1024 * 1024 * 1024);
        ResetWriteMemoryCallbackOffset();

        auto start_time = std::chrono::high_resolution_clock::now();

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

        CURLcode res = curl_easy_perform(curl);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        double bytes_received = 0;

        if (res == CURLE_OK) {
            curl_easy_getinfo(curl, CURLINFO_SIZE_DOWNLOAD, &bytes_received);
            std::cout << "Bytes received: " << bytes_received << " bytes" << std::endl;
        } else {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
        }

        std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;

        if (isFirstQuery) {
            timingFile.open("../timing_results_parquet.txt");
            if (timingFile.is_open()) {
                timingFile << "Query #\t" << queryIndex << "\n";
                timingFile << "Total\t" << std::fixed << std::setprecision(3) << elapsed_seconds.count() << "\n";
                timingFile << "Bytes\t" << std::fixed << std::setprecision(0) << bytes_received << "\n";
                timingFile.close();
                isFirstQuery = false;
            }
        } else {
            std::ifstream inputFile("../timing_results_parquet.txt");
            std::string line;
            std::vector<std::string> lines;
            if (inputFile.is_open()) {
                while (std::getline(inputFile, line)) {
                    lines.push_back(line);
                }
                inputFile.close();

                timingFile.open("../timing_results_parquet.txt");
                if (timingFile.is_open()) {
                    if (lines.size() >= 3) {
                        lines[0] += "\t" + std::to_string(queryIndex);
                        lines[1] += "\t" + std::to_string(elapsed_seconds.count());
                        lines[2] += "\t" + std::to_string(static_cast<long>(bytes_received));
                    }
                    for (const auto& modified_line : lines) {
                        timingFile << modified_line << "\n";
                    }
                    timingFile.close();
                }
            }
        }

        queryIndex++;
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    std::cout << "All queries processed." << std::endl;

    return 0;
}

