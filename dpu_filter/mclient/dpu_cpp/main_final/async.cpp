#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// Callback function for writing data received from the server into memory
size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto& memory = *static_cast<std::vector<char>*>(userp);
    size_t totalSize = size * nmemb;
    static size_t currentOffset = 0; // Maintains the current offset where data is to be written
    
    if (currentOffset + totalSize > memory.size()) {
        std::cerr << "Buffer overflow detected: incoming data exceeds allocated buffer size." << std::endl;
        return 0; // Return 0 to signal an error to libcurl and stop the transfer
    }

    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, memory.begin() + currentOffset);
    currentOffset += totalSize;
    return totalSize;
}

// Function to download a file using HTTP GET without saving it to memory
bool DownloadFileAsync(const std::string& url, std::vector<char>& data) {
    CURLM* multi_handle = curl_multi_init();
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl initialization failed" << std::endl;
        return false;
    }
    data.resize(746619286); // Pre-allocate space to avoid reallocations

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 1024*1024); // Set buffer size to 1MB
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects

    curl_multi_add_handle(multi_handle, curl);
    int still_running = 1; // Assume at least one handle is running

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    do {
        CURLMcode mc = curl_multi_perform(multi_handle, &still_running);
        if (mc != CURLM_OK) {
            std::cerr << "curl_multi_perform() failed." << std::endl;
            break;
        }

        int numfds;
        mc = curl_multi_wait(multi_handle, NULL, 0, 1000, &numfds);
        if (mc != CURLM_OK) {
            std::cerr << "curl_multi_wait() failed." << std::endl;
            break;
        }
    } while (still_running);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Clean up
    curl_multi_remove_handle(multi_handle, curl);
    curl_easy_cleanup(curl);
    curl_multi_cleanup(multi_handle);

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to fetch data asynchronously: " << elapsed.count() << " seconds." << std::endl;

    return true;
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    std::string url = "http://10.10.10.18/dataStat_1000000.parquet";
    std::vector<char> memoryData;

    if (DownloadFileAsync(url, memoryData)) {
        std::cout << "Data fetched successfully. Size: " << memoryData.size() << " bytes." << std::endl;
    } else {
        std::cerr << "Data fetch failed" << std::endl;
    }

    curl_global_cleanup();
    return 0;
}
