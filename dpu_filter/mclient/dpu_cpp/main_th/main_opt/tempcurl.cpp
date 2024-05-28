#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm> // For std::transform
#include <cctype>    // For std::tolower
#include <locale>    // For std::locale

size_t header_callback(char *buffer, size_t size, size_t nitems, void *userdata) {
    std::string header(buffer, size * nitems);
    std::locale loc;  // Use the C++ locale to handle characters properly

    // Transform header to lowercase
    std::transform(header.begin(), header.end(), header.begin(),
                   [&loc](char c) { return std::tolower(c, loc); }); // Using locale

    // Find and parse the Content-Length header
    std::string content_length_key = "content-length: ";
    auto pos = header.find(content_length_key);
    if (pos != std::string::npos) {
        std::string content_length_str = header.substr(pos + content_length_key.length());
        *static_cast<size_t*>(userdata) = std::stoll(content_length_str);
    }
    return nitems * size;
}

bool GetContentSize(const std::string& url, size_t& content_size) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "CURL initialization failed." << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &content_size);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // Perform a HEAD request

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }

    return true;
}


// Callback function for writing data received from the server into memory
size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto& memory = *static_cast<std::vector<char>*>(userp);
    size_t totalSize = size * nmemb;
    static size_t currentOffset = 0; // Maintains the current offset where data is to be written

    // Check if the incoming data fits in the remaining buffer space
    /*if (currentOffset + totalSize > memory.size()) {
        std::cerr << "Buffer overflow detected: incoming data exceeds allocated buffer size." << std::endl;
        return 0; // Return 0 to signal an error to libcurl and stop the transfer
    }*/

    // Copy the received data into the vector at the current offset
    std::copy(static_cast<char*>(contents), static_cast<char*>(contents) + totalSize, memory.begin() + currentOffset);
    currentOffset += totalSize; // Update the offset
    //std::cout << "Received " << totalSize << " bytes this call." << std::endl; // Print the amount of data received in this chunk

    return totalSize;
}



// Function to download a file using HTTP GET without saving it to memory
bool DownloadFileAsync(const std::string& url, std::vector<char>& data) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl initialization failed" << std::endl;
        return false;
    }
    //data.resize(746619286);

    // Set URL and other options
    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 1024*1024*2L); 
    
   

    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to fetch data: " << elapsed.count() << " seconds." << std::endl;

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

    std::string url = "http://10.10.10.18/dataStat_1000000.parquet"; // Change to your actual URL
    size_t content_size = 0;
    if (GetContentSize(url, content_size)) {
        std::cout << "Content size: " << content_size << " bytes." << std::endl;
        std::vector<char> memoryData;
        memoryData.resize(content_size);  // Reserve the exact amount of data

        if (DownloadFileAsync(url, memoryData)) {
            std::cout << "Data fetched successfully. Size: " << memoryData.size() << " bytes." << std::endl;
        } else {
            std::cerr << "Data fetch failed" << std::endl;
        }
    } else {
        std::cerr << "Failed to retrieve content size." << std::endl;
    }

    curl_global_cleanup();
    return 0;
}