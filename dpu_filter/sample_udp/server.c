#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define UDP_PORT 12345
#define BUFFER_SIZE 1024
#define INTERFACE_NAME "ens785f0np0"
#define SERVER_IP "10.10.10.111"
#define RESPONSE_MESSAGE "Hello client!"

int main() {
    int sockfd;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);  // Initialize clientAddrLen
    char buffer[BUFFER_SIZE];

    // Create a UDP socket
    
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Error in socket");
        exit(1);
    }

    // Set the network interface for socket binding
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, INTERFACE_NAME, strlen(INTERFACE_NAME)) < 0) {
        perror("Error in setsockopt");
        exit(1);
    }

    memset(&serverAddr, 0, sizeof(serverAddr));

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(UDP_PORT);
    serverAddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    // Bind the socket to the specified IP address and port
    if (bind(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Error in bind");
        exit(1);
    }

    printf("Server listening on %s:%d\n", SERVER_IP, UDP_PORT);

    while (1) {
        memset(buffer, 0, BUFFER_SIZE);

        // Receive incoming messages
        ssize_t numBytes = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&clientAddr, &clientAddrLen);
        if (numBytes < 0) {
            perror("Error in recvfrom");
            exit(1);
        }

        char clientIP[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(clientAddr.sin_addr), clientIP, INET_ADDRSTRLEN);

        printf("Received message: '%s' from %s:%d\n", buffer, clientIP, ntohs(clientAddr.sin_port));

        // Send response back to the client
        serverAddr.sin_addr.s_addr = clientAddr.sin_addr.s_addr; // Update the server address to the client's address
        numBytes = sendto(sockfd, RESPONSE_MESSAGE, strlen(RESPONSE_MESSAGE), 0, (struct sockaddr *)&serverAddr, sizeof(serverAddr));
        if (numBytes < 0) {
            perror("Error in sendto");
            exit(1);
        }

        printf("Sent response: '%s' to %s:%d\n", RESPONSE_MESSAGE, clientIP, ntohs(clientAddr.sin_port));
    }

    return 0;
}