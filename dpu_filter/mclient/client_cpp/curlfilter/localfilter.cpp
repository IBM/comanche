#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// Callback function for writing data received from the server into memory
size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, std::vector<char>* memory) {
    size_t totalSize = size * nmemb;
    memory->insert(memory->end(), (char*)contents, (char*)contents + totalSize);
    return totalSize;
}

// Function to download a file using HTTP GET and save it to memory
bool DownloadFileAsync(const std::string& url, std::vector<char>& data) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl initialization failed" << std::endl;
        return false;
    }

    // Pre-allocate the vector to hold the data, 713MB
    data.resize(713LL * 1024 * 1024);

    // Set URL and other options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects

    // Measure the time taken for curl_easy_perform
    auto performStart = std::chrono::high_resolution_clock::now();

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    auto performEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> performElapsed = performEnd - performStart;
    std::cout << "Time taken by curl_easy_perform: " << performElapsed.count() << " seconds." << std::endl;

    // Check for errors
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    // Clean up
    curl_easy_cleanup(curl);
    return true;
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::string url = "http://10.10.10.18/dataStat_1000000.parquet";
    std::vector<char> memoryData; 

    if (DownloadFileAsync(url, memoryData)) {
        std::cout << "Data fetched successfully." << std::endl;
        std::cout << "Actual data size received: " << memoryData.size() << " bytes." << std::endl;
    } else {
        std::cerr << "Data fetch failed" << std::endl;
    }

    curl_global_cleanup();
    return 0;
}
