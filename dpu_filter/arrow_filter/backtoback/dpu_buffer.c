#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define CLIENT_TCP_PORT 8888
#define SERVER_TCP_PORT 8888
#define CLIENT_IP "192.168.0.154" // Client's IP address
#define SERVER_IP "192.168.0.42"  // Server's IP address
#define BUFFER_SIZE 1024
#define FILE_DATA_BUFFER_SIZE (1024 * 1024) // 1024KB

int main() {
    int serverSock, clientSock;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr); // Initialize clientAddrLen
    char buffer[BUFFER_SIZE];
     // Pre-allocate the file data buffer
    unsigned char *fileDataBuffer = malloc(FILE_DATA_BUFFER_SIZE);
    if (fileDataBuffer == NULL) {
        perror("Error allocating memory for file data buffer");
        exit(1);
    }

    size_t bufferCapacity = FILE_DATA_BUFFER_SIZE;

    size_t totalBytesReceived = 0;

    // Create a TCP socket to wait for the client connection
    serverSock = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSock < 0) {
        perror("Error creating server socket");
        exit(1);
    }

    memset(&serverAddr, 0, sizeof(serverAddr));

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(CLIENT_TCP_PORT);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket to the specified IP address and port
    if (bind(serverSock, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Error in bind");
        exit(1);
    }

    // Listen for incoming client connections
    if (listen(serverSock, 5) < 0) {
        perror("Error in listen");
        exit(1);
    }

    printf("Middle machine waiting for client connection.\n");

    while (1) {
        // Accept incoming client connection
        clientSock = accept(serverSock, (struct sockaddr *)&clientAddr, &clientAddrLen);
        if (clientSock < 0) {
            perror("Error in accept");
            continue;  // Continue waiting for the next client
        }

        printf("Connected to the client.\n");

        // Receive the filename from the client
        memset(buffer, 0, BUFFER_SIZE);
        int bytesReadFromClient = recv(clientSock, buffer, BUFFER_SIZE, 0);
        if (bytesReadFromClient <= 0) {
            perror("Error receiving filename from client");
            close(clientSock);
            continue;  // Continue waiting for the next client
        }

        // Create a TCP socket to connect to the server
        int serverSock = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSock < 0) {
            perror("Error creating server socket");
            close(clientSock);
            continue;  // Continue waiting for the next client
        }

        memset(&serverAddr, 0, sizeof(serverAddr));

        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(SERVER_TCP_PORT);
        if (inet_aton(SERVER_IP, &serverAddr.sin_addr) == 0) {
            perror("Error in inet_aton for server");
            close(clientSock);
            close(serverSock);
            continue;  // Continue waiting for the next client
        }

        // Connect to the server
        if (connect(serverSock, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
            perror("Error connecting to server");
            close(clientSock);
            close(serverSock);
            continue;  // Continue waiting for the next client
        }

        printf("Connected to the server.\n");

        // Send the filename to the server
        int bytesSentToServer = send(serverSock, buffer, bytesReadFromClient, 0);
        if (bytesSentToServer <= 0) {
            perror("Error sending filename to server");
            close(clientSock);
            close(serverSock);
            continue;  // Continue waiting for the next client
        }

// Receive data from the server
while (1) {
    int bytesReadFromServer = recv(serverSock, buffer, BUFFER_SIZE, 0);
    if (bytesReadFromServer <= 0) {
        // End of file or error receiving from server
        break;
    }

    // Resize the buffer to accommodate the new data
    size_t newCapacity = totalBytesReceived + bytesReadFromServer;
    if (newCapacity > bufferCapacity) {
        while (newCapacity > bufferCapacity) {
            bufferCapacity *= 2; // Double the buffer capacity
        }
        fileDataBuffer = realloc(fileDataBuffer, bufferCapacity);
        if (fileDataBuffer == NULL) {
            perror("Error reallocating memory for file data buffer");
            break;
        }
    }

    // Copy received data to the buffer
    memcpy(fileDataBuffer + totalBytesReceived, buffer, bytesReadFromServer);
    totalBytesReceived += bytesReadFromServer;
}


        printf("Received the full file from the server.\n");

        // Forward data from the buffer to the client
        size_t bytesSentToClient = 0;
        while (bytesSentToClient < totalBytesReceived) {
            size_t remainingBytesToSend = totalBytesReceived - bytesSentToClient;
            size_t bytesToSend = remainingBytesToSend < BUFFER_SIZE ? remainingBytesToSend : BUFFER_SIZE;

            // Send data to the client
            int bytesSent = send(clientSock, fileDataBuffer + bytesSentToClient, bytesToSend, 0);
            if (bytesSent <= 0) {
                perror("Error sending data to client");
                break;
            }

            bytesSentToClient += bytesSent;
        }

        printf("File forwarded between the client and the server.\n");

        // Clean up
        free(fileDataBuffer);
        totalBytesReceived = 0;

        // Close the sockets
        close(clientSock);
        close(serverSock);
    }

    return 0;
}
