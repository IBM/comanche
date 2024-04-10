#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

const int TCP_PORT = 12345;
const std::string FILE_DIRECTORY = "/mnt/sda4/";
const size_t BUFFER_SIZE = 1024*64; // Corrected to 64KB for demonstration

int main() {
    int sockfd, newsockfd;
    sockaddr_in serverAddr{}, clientAddr{};
    socklen_t clientAddrLen = sizeof(clientAddr);
    char buffer[BUFFER_SIZE];

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error in socket creation" << std::endl;
        exit(1);
    }

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(TCP_PORT);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, reinterpret_cast<struct sockaddr*>(&serverAddr), sizeof(serverAddr)) < 0) {
        std::cerr << "Error in bind" << std::endl;
        exit(1);
    }

    if (listen(sockfd, 5) < 0) {
        std::cerr << "Error in listen" << std::endl;
        exit(1);
    }

    std::cout << "Server listening on port " << TCP_PORT << std::endl;

    while (true) {
        newsockfd = accept(sockfd, reinterpret_cast<struct sockaddr*>(&clientAddr), &clientAddrLen);
        if (newsockfd < 0) {
            std::cerr << "Error in accept" << std::endl;
            continue;
        }

        std::cout << "Accepted connection from " << inet_ntoa(clientAddr.sin_addr) << ":" << ntohs(clientAddr.sin_port) << std::endl;

        ssize_t numBytes = recv(newsockfd, buffer, BUFFER_SIZE - 1, 0);
        if (numBytes < 0) {
            std::cerr << "Error in receiving filename" << std::endl;
            close(newsockfd);
            continue;
        }

        buffer[numBytes] = '\0'; // Null-terminate the received filename
        std::string filepath = FILE_DIRECTORY + std::string(buffer);

        std::ifstream file(filepath, std::ifstream::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filepath << std::endl;
            close(newsockfd);
            continue;
        }

        // Determine the file size
        file.seekg(0, file.end);
        uint64_t fileSize = file.tellg();
        file.seekg(0, file.beg);
        fileSize = htonl(fileSize); // Ensure correct byte order
        // Send the file size to the client
        send(newsockfd, &fileSize, sizeof(fileSize), 0);

        // Send the file data in chunks
        while (!file.eof()) {
            file.read(buffer, BUFFER_SIZE);
            std::streamsize bytesRead = file.gcount();
            if (bytesRead > 0) {
                send(newsockfd, buffer, bytesRead, 0);
            }
        }

        std::cout << "File sent to the client." << std::endl;
        file.close();
        close(newsockfd);
    }

    close(sockfd);
    return 0;
}
