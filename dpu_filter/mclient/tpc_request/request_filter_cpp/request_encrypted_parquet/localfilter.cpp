#include <iostream>
#include <chrono>
#include <string>
#include <curl/curl.h>
#include <vector>
#include <cstring>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/stream_writer.h>

#define MAX_BUFFER_SIZE 1048576 // 1MB buffer size
#define TAG_SIZE 12 // 12 bytes tag size
#define KEY_SIZE 32 // 256 bits
#define IV_SIZE 12 // 96 bits

void handleErrors(const std::string &message) {
    std::cerr << message << std::endl;
    ERR_print_errors_fp(stderr);
    abort();
}

bool decryptChunk(const unsigned char *key, const unsigned char *iv, const unsigned char *ciphertext, int ciphertext_len, const unsigned char *tag, unsigned char *plaintext) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len;
    int plaintext_len;
    bool success = false;

    if (!ctx) {
        handleErrors("Failed to create EVP_CIPHER_CTX");
        return false;
    }

    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) goto cleanup;
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, IV_SIZE, NULL) != 1) goto cleanup;
    if (EVP_DecryptInit_ex(ctx, NULL, NULL, key, iv) != 1) goto cleanup;

    if (EVP_DecryptUpdate(ctx, NULL, &len, NULL, 0) != 1) goto cleanup;  // No AAD

    if (EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len) != 1) goto cleanup;
    plaintext_len = len;

    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, TAG_SIZE, (void *)tag) != 1) goto cleanup;

    if (EVP_DecryptFinal_ex(ctx, plaintext + len, &len) != 1) goto cleanup;
    plaintext_len += len;
    success = true;

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    return success;
}

std::vector<unsigned char> decryptBuffer(const std::vector<unsigned char>& encrypted_data, const unsigned char *key, const unsigned char *iv) {
    size_t bufferSize = MAX_BUFFER_SIZE - TAG_SIZE;
    std::vector<unsigned char> buffer(MAX_BUFFER_SIZE);
    std::vector<unsigned char> ciphertext(bufferSize);
    std::vector<unsigned char> tag(TAG_SIZE);
    std::vector<unsigned char> decrypted_data;

    size_t offset = 0;
    while (offset + MAX_BUFFER_SIZE <= encrypted_data.size()) {
        std::memcpy(ciphertext.data(), encrypted_data.data() + offset, bufferSize);
        std::memcpy(tag.data(), encrypted_data.data() + offset + bufferSize, tag.size());

        std::vector<unsigned char> plaintext(bufferSize);

        if (!decryptChunk(key, iv, ciphertext.data(), bufferSize, tag.data(), plaintext.data())) {
            handleErrors("Decryption failed!");
        }

        decrypted_data.insert(decrypted_data.end(), plaintext.begin(), plaintext.end());
        offset += MAX_BUFFER_SIZE;
    }

    // Handle the last chunk if it is smaller than MAX_BUFFER_SIZE
    size_t lastChunkSize = encrypted_data.size() - offset;
    if (lastChunkSize > 0) {
        if (lastChunkSize > TAG_SIZE) {
            buffer.resize(lastChunkSize);
            ciphertext.resize(lastChunkSize - TAG_SIZE);

            std::memcpy(ciphertext.data(), encrypted_data.data() + offset, lastChunkSize - TAG_SIZE);
            std::memcpy(tag.data(), encrypted_data.data() + offset + lastChunkSize - TAG_SIZE, tag.size());

            std::vector<unsigned char> plaintext(lastChunkSize - TAG_SIZE);

            if (!decryptChunk(key, iv, ciphertext.data(), lastChunkSize - TAG_SIZE, tag.data(), plaintext.data())) {
                handleErrors("Decryption failed!");
            }

            decrypted_data.insert(decrypted_data.end(), plaintext.begin(), plaintext.end());
        } else {
            handleErrors("Error: Last chunk is too small to contain valid data and tag.");
        }
    }

    return decrypted_data;
}

// Function to write the response data to a vector
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::vector<unsigned char>* s) {
    size_t totalSize = size * nmemb;
    s->insert(s->end(), (unsigned char*)contents, (unsigned char*)contents + totalSize);
    return totalSize;
}

int main() {
    const std::string url = "http://10.10.10.20:8080/data";
    const std::string payload = R"({
        "bucket": "encrypted",
        "key": "enc_inventory.parquet",
        "sql": "SELECT inv_item_sk inv_quantity_on_hand FROM inventory WHERE inv_date_sk <= 2450997"
    })";

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return 1;
    }

    std::vector<unsigned char> response_data;
    auto start_time = std::chrono::high_resolution_clock::now();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 2 * 1024 * 1024); // Increased buffer size
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 120L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 60L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L); // Set a timeout for the entire request
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L); // Set a connection timeout

    CURLcode res = curl_easy_perform(curl);

    auto fetch_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fetch_seconds = fetch_time - start_time;

    std::cout << "Time taken to fetch: " << fetch_seconds.count() << " seconds" << std::endl;

    if (res != CURLE_OK) {
        std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
    } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code == 200) {
            std::cout << "Success: Data received." << std::endl;

            const unsigned char key[KEY_SIZE] = {0}; // 256-bit key (all zeros, replace with actual key)
            const unsigned char iv[IV_SIZE] = {0};  // 96-bit IV (all zeros, replace with actual IV)

            std::vector<unsigned char> decrypted_data = decryptBuffer(response_data, key, iv);
            std::cout << "Decryption completed." << std::endl;

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;

            std::cout << "Time taken with decrypt: " << elapsed_seconds.count() << " seconds" << std::endl;

            // Read and print the Parquet file content
            auto buffer = std::make_shared<arrow::Buffer>(decrypted_data.data(), decrypted_data.size());
            auto input = std::make_shared<arrow::io::BufferReader>(buffer);
            std::unique_ptr<parquet::arrow::FileReader> reader;
            PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &reader));

            std::shared_ptr<arrow::Table> table;
            PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

            std::cout << "Table contents:" << std::endl;
            std::cout << table->ToString() << std::endl;
            std::cout << "Total number of rows in the filtered table: " << table->num_rows() << std::endl;

            auto print_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> print_seconds = print_time - start_time;

            std::cout << "Time taken to print: " << print_seconds.count() << " seconds" << std::endl;

        } else {
            std::cerr << "HTTP Error: " << http_code << std::endl;
        }
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    return 0;
}
