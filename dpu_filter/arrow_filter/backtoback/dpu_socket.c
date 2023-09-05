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

int main() {
    int serverSock, clientSock;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr); // Initialize clientAddrLen
    char buffer[BUFFER_SIZE];

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

        // Receive data from the server and forward it to the client
        while (1) {
            int bytesReadFromServer = recv(serverSock, buffer, BUFFER_SIZE, 0);
            if (bytesReadFromServer <= 0) {
                // End of file or error receiving from server
                break;
            }

            // Send data to the client
            int bytesSentToClient = send(clientSock, buffer, bytesReadFromServer, 0);
            if (bytesSentToClient <= 0) {
                perror("Error sending data to client");
                break;
            }
        }

        printf("File forwarded between the client and the server.\n");

        // Close the sockets
        close(clientSock);
        close(serverSock);
    }

    return 0;
}
