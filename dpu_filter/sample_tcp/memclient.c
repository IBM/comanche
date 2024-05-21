#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>

#define SERVER_IP "10.10.10.18"
#define TCP_PORT 12345
#define BUFFER_SIZE (1024*1024*2) // 2MB buffer size

int main() {
    int sockfd;
    struct sockaddr_in serverAddr;
    char *fileData; // Dynamic array to store the file directly
    char filename[BUFFER_SIZE];
    char buffer[BUFFER_SIZE]; 
    clock_t start_time, end_time;
    long fileSize, bytesReceivedTotal = 0;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Error in socket creation");
        exit(1);
    }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(TCP_PORT);
    serverAddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    if (connect(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Error in connect");
        exit(1);
    }

    printf("Enter the filename to request: ");
    scanf("%s", filename); // Reads the filename from user input

    send(sockfd, filename, strlen(filename), 0);

    uint32_t netFileSize;
    recv(sockfd, &netFileSize, sizeof(netFileSize), 0);
    fileSize = ntohl(netFileSize); // Converts network byte order to host byte order
    printf("Receiving file of size: %ld bytes\n", fileSize);

    fileData = malloc(fileSize); // Allocate memory for the entire file
    if (!fileData) {
        perror("Failed to allocate memory");
        exit(1);
    }

    start_time = clock(); // Record the start time

    // Receiving data directly into the pre-allocated buffer
    while (bytesReceivedTotal < fileSize) {
       // int bytesReceived = recv(sockfd, buffer, BUFFER_SIZE, 0); 
        int bytesReceived = recv(sockfd, fileData + bytesReceivedTotal, fileSize - bytesReceivedTotal, 0);
        if (bytesReceived <= 0) break; // Error or connection closed
        bytesReceivedTotal += bytesReceived;
    }

    end_time = clock(); // Record the end time

    // Calculate and print the elapsed time in milliseconds
    double elapsed_time = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;
    printf("File received from the server in %.2f milliseconds.\n", elapsed_time);

    if (bytesReceivedTotal != fileSize) {
        printf("Error: File received partially. Expected %ld bytes, received %ld bytes.\n", fileSize, bytesReceivedTotal);
    } else {
        printf("File received successfully.\n");
    }

    free(fileData); // Don't forget to free the allocated memory
    close(sockfd);

    return 0;
}
