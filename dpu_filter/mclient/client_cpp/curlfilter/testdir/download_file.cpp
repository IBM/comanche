#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>  // Include this header for std::chrono

struct DownloadData {
    std::vector<char> data;
};

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    DownloadData* downloadData = static_cast<DownloadData*>(userp);
    size_t current_size = downloadData->data.size();
    downloadData->data.resize(current_size + totalSize);
    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, downloadData->data.begin() + current_size);
    return totalSize;
}

bool DownloadFile(const std::string& url, DownloadData& downloadData) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl initialization failed" << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &downloadData);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 2 * 1024 * 1024); // Increased buffer size
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

    // Enable HTTP/2
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_cleanup(curl);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <URL>" << std::endl;
        return 1;
    }

    std::string url = argv[1];

    curl_global_init(CURL_GLOBAL_DEFAULT);

    DownloadData downloadData;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (DownloadFile(url, downloadData)) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Data fetched successfully. Total time taken to fetch data: " << elapsed.count() << " seconds." << std::endl;
    } else {
        std::cerr << "Data fetch failed" << std::endl;
    }

    curl_global_cleanup();

    return 0;
}

