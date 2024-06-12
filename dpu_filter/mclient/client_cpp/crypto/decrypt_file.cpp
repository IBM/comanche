#include <openssl/evp.h>
#include <openssl/err.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

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

void decryptFile(const std::string& inputFilePath, const std::string& outputFilePath) {
    const unsigned char key[KEY_SIZE] = {0}; // 256-bit key (all zeros)
    const unsigned char iv[IV_SIZE] = {0};  // 96-bit IV (all zeros)

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

int main() {
    const std::string encryptedFilePath = "/mnt/sda4/encrypted_600000.parquet";
    const std::string decryptedFilePath = "decrypted_file.parquet";

    decryptFile(encryptedFilePath, decryptedFilePath);

    std::cout << "Decryption completed." << std::endl;

    return 0;
}

