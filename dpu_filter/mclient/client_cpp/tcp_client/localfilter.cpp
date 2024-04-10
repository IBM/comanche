#include <pistache/endpoint.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <vector>
#include <aws/s3/model/HeadObjectRequest.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <chrono>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h> // Include this header
#include "/home/ubuntu/json/include/nlohmann/json.hpp"
#include "SQLParser.h"
#include <future>
#include <mutex>
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fstream>


#include <iostream>
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>

#define SERVER_IP "10.10.10.18"
#define TCP_PORT 12345
#define BUFFER_SIZE (1024*64) // 2MB buffer size

int main() {
    int sockfd;
    sockaddr_in serverAddr{};
    std::vector<char> fileData;
    std::string filename;
    long fileSize, bytesReceivedTotal = 0;
    char buffer[BUFFER_SIZE]; // Temporary buffer for receiving data


    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error in socket creation" << std::endl;
        exit(1);
    }

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(TCP_PORT);
    serverAddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    if (connect(sockfd, reinterpret_cast<struct sockaddr*>(&serverAddr), sizeof(serverAddr)) < 0) {
        std::cerr << "Error in connect" << std::endl;
        exit(1);
    }

    std::cout << "Enter the filename to request: ";
    std::cin >> filename;

    send(sockfd, filename.c_str(), filename.length(), 0);

    uint32_t netFileSize;
    recv(sockfd, &netFileSize, sizeof(netFileSize), 0);
    fileSize = ntohl(netFileSize);
    std::cout << "Receiving file of size: " << fileSize << " bytes\n";

    fileData.resize(fileSize);

    auto start_time = std::chrono::high_resolution_clock::now();

    while (bytesReceivedTotal < fileSize) {
        int bytesReceived = recv(sockfd, buffer, BUFFER_SIZE, 0); 
        //int bytesReceived = recv(sockfd, &fileData[bytesReceivedTotal], fileSize - bytesReceivedTotal, 0);
        if (bytesReceived <= 0) break; // Error or connection closed
        bytesReceivedTotal += bytesReceived;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
    std::cout << "File received from the server in " << elapsed_time.count() << " milliseconds.\n";

    if (bytesReceivedTotal != fileSize) {
        std::cout << "Error: File received partially. Expected " << fileSize << " bytes, received " << bytesReceivedTotal << " bytes.\n";
    } else {
        std::cout << "File received successfully.\n";
    }

    close(sockfd);

    return 0;
}