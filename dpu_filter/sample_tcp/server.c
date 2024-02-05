#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h> // for sleep function

#define TCP_PORT 12345
#define FILE_DIRECTORY "/mnt/sda4/"
#define BUFFER_SIZE 1024
#define SLEEP_INTERVAL 1 // Sleep interval in seconds
#define MAX_DATA_SIZE (64 * 1024) // Maximum data size before sleeping

int main() {
    int sockfd, newsockfd;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr); // Initialize clientAddrLen
    char buffer[BUFFER_SIZE];
    size_t totalBytesSent = 0;

    // Create a TCP socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Error in socket");
        exit(1);
    }

    memset(&serverAddr, 0, sizeof(serverAddr));

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(TCP_PORT);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket to the specified IP address and port
    if (bind(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Error in bind");
        exit(1);
    }

    // Listen for incoming connections
    if (listen(sockfd, 5) < 0) {
        perror("Error in listen");
        exit(1);
    }

    printf("Server listening on port %d\n", TCP_PORT);

    while (1) {
        // Accept incoming connections
        newsockfd = accept(sockfd, (struct sockaddr *)&clientAddr, &clientAddrLen);
        if (newsockfd < 0) {
            perror("Error in accept");
            exit(1);
        }

        printf("Accepted connection from %s:%d\n", inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));

        // Receive the filename requested by the client
        memset(buffer, 0, BUFFER_SIZE);
        ssize_t numBytes = recv(newsockfd, buffer, BUFFER_SIZE - 1, 0);
        if (numBytes < 0) {
            perror("Error in receiving filename");
            close(newsockfd);
            continue;
        }

        // Null-terminate the received filename
        buffer[numBytes] = '\0';

        // Create the complete path to the file
        char filepath[BUFFER_SIZE];
        snprintf(filepath, sizeof(filepath), "%s%s", FILE_DIRECTORY, buffer);

        // Open the file for reading
        FILE *file = fopen(filepath, "rb");
        if (file == NULL) {
            perror("Error opening file");
            close(newsockfd);
            continue;
        }

        totalBytesSent = 0; // Reset the byte count

        // Send file contents to the client
        while (1) {
            // Read data from the file
            int bytesRead = fread(buffer, 1, BUFFER_SIZE, file);
            if (bytesRead <= 0) {
                // End of file or error reading
                break;
            }

            // Send data to the client
            int bytesSent = send(newsockfd, buffer, bytesRead, 0);
            if (bytesSent <= 0) {
                perror("Error in sending file");
                break;
            }

            /*totalBytesSent += bytesSent;

            // Check if the total data size sent exceeds 64KB and introduce a 1-second delay
            if (totalBytesSent >= MAX_DATA_SIZE) {
                sleep(SLEEP_INTERVAL); // Introduce a 1-second delay
                totalBytesSent = 0;    // Reset the byte count
            }*/
        }

        printf("File sent to the client.\n");

        // Close the file and the new socket
        fclose(file);
        close(newsockfd);
    }

    // Close the listening socket
    close(sockfd);

    return 0;
}
