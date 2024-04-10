#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define TCP_PORT 12345
#define FILE_DIRECTORY "/path/to/directory/" // Update with your file directory
#define BUFFER_SIZE 1024

int main() {
    int sockfd, newsockfd;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    char buffer[BUFFER_SIZE];

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Error in socket creation");
        exit(1);
    }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(TCP_PORT);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Error in bind");
        exit(1);
    }

    if (listen(sockfd, 5) < 0) {
        perror("Error in listen");
        exit(1);
    }

    printf("Server listening on port %d\n", TCP_PORT);

    while (1) {
        newsockfd = accept(sockfd, (struct sockaddr *)&clientAddr, &clientAddrLen);
        if (newsockfd < 0) {
            perror("Error in accept");
            continue;
        }

        printf("Accepted connection from %s:%d\n", inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));

        memset(buffer, 0, BUFFER_SIZE);
        ssize_t numBytes = recv(newsockfd, buffer, BUFFER_SIZE - 1, 0);
        if (numBytes < 0) {
            perror("Error receiving filename");
            close(newsockfd);
            continue;
        }

        buffer[numBytes] = '\0';
        printf("Requested file: %s\n", buffer);

        char filepath[256];
        snprintf(filepath, sizeof(filepath), "%s%s", FILE_DIRECTORY, buffer);

        FILE *file = fopen(filepath, "rb");
        if (!file) {
            perror("Error opening file");
            close(newsockfd);
            continue;
        }

        fseek(file, 0L, SEEK_END);
        long fileSize = ftell(file);
        rewind(file);

        uint32_t netFileSize = htonl(fileSize);
        send(newsockfd, &netFileSize, sizeof(netFileSize), 0);

        while (!feof(file)) {
            int bytesRead = fread(buffer, 1, BUFFER_SIZE, file);
            if (bytesRead > 0) {
                send(newsockfd, buffer, bytesRead, 0);
            }
        }

        printf("File transmission complete.\n");
        fclose(file);
        close(newsockfd);
    }

    close(sockfd);
    return 0;
}

