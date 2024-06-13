#include <iostream>
#include <fstream>
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

    if (!ctx) return false;

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

void decryptFile(const std::string& inputFilePath, const std::string& outputFilePath, const unsigned char *key, const unsigned char *iv) {
    std::ifstream inputFile(inputFilePath, std::ios::binary);
    std::ofstream outputFile(outputFilePath, std::ios::binary);

    if (!inputFile.is_open() || !outputFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    size_t bufferSize = MAX_BUFFER_SIZE - TAG_SIZE;
    std::vector<unsigned char> buffer(MAX_BUFFER_SIZE);
    std::vector<unsigned char> ciphertext(bufferSize);
    std::vector<unsigned char> tag(TAG_SIZE);

    while (inputFile.read(reinterpret_cast<char*>(buffer.data()), MAX_BUFFER_SIZE)) {
        std::memcpy(ciphertext.data(), buffer.data(), bufferSize);
        std::memcpy(tag.data(), buffer.data() + bufferSize, tag.size());

        std::vector<unsigned char> plaintext(bufferSize);

        if (!decryptChunk(key, iv, ciphertext.data(), bufferSize, tag.data(), plaintext.data())) {
            handleErrors("Decryption failed!");
        }

        outputFile.write(reinterpret_cast<char*>(plaintext.data()), plaintext.size());
    }

    // Handle the last chunk if it is smaller than MAX_BUFFER_SIZE
    std::streamsize lastChunkSize = inputFile.gcount();
    if (lastChunkSize > 0) {
        if (lastChunkSize > TAG_SIZE) {
            buffer.resize(lastChunkSize);
            ciphertext.resize(lastChunkSize - TAG_SIZE);

            std::memcpy(ciphertext.data(), buffer.data(), lastChunkSize - TAG_SIZE);
            std::memcpy(tag.data(), buffer.data() + lastChunkSize - TAG_SIZE, tag.size());

            std::vector<unsigned char> plaintext(lastChunkSize - TAG_SIZE);

            if (!decryptChunk(key, iv, ciphertext.data(), lastChunkSize - TAG_SIZE, tag.data(), plaintext.data())) {
                handleErrors("Decryption failed!");
            }

            outputFile.write(reinterpret_cast<char*>(plaintext.data()), plaintext.size());
        } else {
            handleErrors("Error: Last chunk is too small to contain valid data and tag.");
        }
    }

    inputFile.close();
    outputFile.close();
}

// Function to write the response data to a string
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t totalSize = size * nmemb;
    s->append((char*)contents, totalSize);
    return totalSize;
}

int main() {
    const std::string url = "http://10.10.10.20:8080/data";
    const std::string payload = R"({
        "bucket": "mycsvbucket",
        "key": "sampledata/dataStat_1000000.parquet",
        "sql": "SELECT * FROM s3object WHERE Age > 60"
    })";

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return 1;
    }

    std::string response_string;
    auto start_time = std::chrono::high_resolution_clock::now();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);

    CURLcode res = curl_easy_perform(curl);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    if (res != CURLE_OK) {
        std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
    } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code == 200) {
            const std::string encryptedFilePath = "encrypted_output.parquet";
            const std::string decryptedFilePath = "decrypted_file.parquet";

            std::ofstream outfile(encryptedFilePath, std::ios::binary);
            outfile.write(response_string.c_str(), response_string.size());
            outfile.close();
            std::cout << "Success: Encrypted file saved as " << encryptedFilePath << std::endl;

            const unsigned char key[KEY_SIZE] = {0}; // 256-bit key (all zeros)
            const unsigned char iv[IV_SIZE] = {0};  // 96-bit IV (all zeros)

            decryptFile(encryptedFilePath, decryptedFilePath, key, iv);
            std::cout << "Decryption completed." << std::endl;

            // Read and print the Parquet file content
            std::shared_ptr<arrow::io::ReadableFile> infile;
            PARQUET_ASSIGN_OR_THROW(
                infile, arrow::io::ReadableFile::Open(decryptedFilePath));
            std::unique_ptr<parquet::arrow::FileReader> reader;
            PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

            std::shared_ptr<arrow::Table> table;
            PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

            std::cout << "Table contents:" << std::endl;
            std::cout << table->ToString() << std::endl;
        } else {
            std::cerr << "HTTP Error: " << http_code << std::endl;
        }
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
