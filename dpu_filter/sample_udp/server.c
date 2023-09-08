#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#define UDP_PORT 12345
#define BUFFER_SIZE 1024
#define FILE_DIRECTORY "/mnt/ssd/"

int main() {
    int sockfd;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    char buffer[BUFFER_SIZE];

    // Create a UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Error in socket");
        exit(1);
    }

    memset(&serverAddr, 0, sizeof(serverAddr));

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(UDP_PORT);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket to the specified IP address and port
    if (bind(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Error in bind");
        exit(1);
    }

    printf("Server listening on port %d\n", UDP_PORT);

    while (1) {
        // Receive filename from the client
        memset(buffer, 0, BUFFER_SIZE);
        ssize_t numBytes = recvfrom(sockfd, buffer, BUFFER_SIZE - 1, 0, (struct sockaddr *)&clientAddr, &clientAddrLen);
        if (numBytes < 0) {
            perror("Error receiving filename");
            continue;
        }

        // Construct the complete path to the file
        char filepath[BUFFER_SIZE];
        snprintf(filepath, sizeof(filepath), "%s%s", FILE_DIRECTORY, buffer);

        // Open the file for reading
        FILE *file = fopen(filepath, "rb");
        if (file == NULL) {
            perror("Error opening file");
            // Send an error message back to the client
            const char *error_msg = "File not found or unable to open";
            sendto(sockfd, error_msg, strlen(error_msg), 0, (struct sockaddr *)&clientAddr, clientAddrLen);
            continue;
        }

        // Read and send file contents in chunks
        while (1) {
            // Read data from the file
            int bytesRead = fread(buffer, 1, BUFFER_SIZE, file);
            if (bytesRead <= 0) {
                // End of file or error reading
                break;
            }

            // Send data to the client
            int bytesSent = sendto(sockfd, buffer, bytesRead, 0, (struct sockaddr *)&clientAddr, clientAddrLen);
            if (bytesSent <= 0) {
                perror("Error sending file content");
                break;
            }
        }

        printf("File sent to the client.\n");

        // Close the file
        fclose(file);
    }

    // Close the listening socket
    close(sockfd);

    return 0;
}
