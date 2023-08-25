#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define UDP_PORT 12345
#define SERVER_IP "10.10.10.111"
#define CLIENT_IP "10.10.10.112"
#define MESSAGE "test"
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in serverAddr, clientAddr;
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

    memset(&clientAddr, 0, sizeof(clientAddr));

    clientAddr.sin_family = AF_INET;
    clientAddr.sin_port = htons(UDP_PORT);
    if (inet_aton(CLIENT_IP, &clientAddr.sin_addr) == 0) {
        perror("Error in inet_aton");
        exit(1);
    }

    // Bind the socket to the specified IP address and port
    if (bind(sockfd, (struct sockaddr *)&clientAddr, sizeof(clientAddr)) < 0) {
        perror("Error in bind");
        exit(1);
    }

    printf("Client sending message: '%s' to %s:%d\n", MESSAGE, SERVER_IP, UDP_PORT);

    // Send message to the server
    ssize_t numBytes = sendto(sockfd, MESSAGE, strlen(MESSAGE), 0, (struct sockaddr *)&serverAddr, sizeof(serverAddr));
    if (numBytes < 0) {
        perror("Error in sendto");
        exit(1);
    }

    memset(buffer, 0, BUFFER_SIZE);

    // Receive response from the server
    numBytes = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, NULL, NULL);
    if (numBytes < 0) {
        perror("Error in recvfrom");
        exit(1);
    }

    printf("Received response: '%s'\n", buffer);

    return 0;
}