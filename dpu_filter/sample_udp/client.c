#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define UDP_PORT 12345
#define SERVER_IP "10.10.10.111" // Replace with the server's IP address
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in serverAddr;
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
    if (inet_aton(SERVER_IP, &serverAddr.sin_addr) == 0) {
        perror("Error in inet_aton");
        exit(1);
    }

    // Get the filename from the user
    printf("Enter the filename you want to request: ");
    scanf("%s", buffer);

    // Send the filename to the server
    int bytesSent = sendto(sockfd, buffer, strlen(buffer), 0, (struct sockaddr *)&serverAddr, sizeof(serverAddr));
    if (bytesSent <= 0) {
        perror("Error in sending filename");
        close(sockfd);
        exit(1);
    }

    // Open the file for writing
    FILE *file = fopen(buffer, "wb");
    if (file == NULL) {
        perror("Error opening file");
        close(sockfd);
        exit(1);
    }

    // Receive file contents from the server
    while (1) {
        // Receive data from the server
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int bytesRead = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&clientAddr, &clientAddrLen);
        if (bytesRead <= 0) {
            // End of file or error receiving
            break;
        }

        // Write data to the file
        int bytesWritten = fwrite(buffer, 1, bytesRead, file);
        if (bytesWritten < bytesRead) {
            perror("Error writing to file");
            break;
        }
    }

    printf("File received from the server.\n");

    // Close the file and the socket
    fclose(file);
    close(sockfd);

    return 0;
}
