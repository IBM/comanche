#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "process_buffer.h"
#include <sys/time.h>

#define CLIENT_TCP_PORT 8888
#define SERVER_TCP_PORT 8888
#define CLIENT_IP "192.168.0.154" // Client's IP address
#define SERVER_IP "192.168.0.42"  // Server's IP address
#define BUFFER_SIZE (1024*1024)
#define FILE_DATA_BUFFER_SIZE (1024 * 1024 * 1024) // 1024KB

long long get_elapsed_milliseconds(struct timeval start_time, struct timeval end_time) {
    return (end_time.tv_sec - start_time.tv_sec) * 1000LL +
           (end_time.tv_usec - start_time.tv_usec) / 1000LL;
}

int main() {
    int serverSock, clientSock;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr); // Initialize clientAddrLen
    struct timeval fetch_start;
    struct timeval fetch_end;
    struct timeval filter_start;
    struct timeval filter_end;
    struct timeval all_start;
    struct timeval all_end;  
    // Pre-allocate the file data buffer
    unsigned char *fileDataBuffer = malloc(FILE_DATA_BUFFER_SIZE);
    if (fileDataBuffer == NULL) {
        perror("Error allocating memory for file data buffer");
        exit(1);
    }

    unsigned char *filteredBuffer;
   
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
        char buffer[BUFFER_SIZE];
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
        
        gettimeofday(&fetch_start, NULL);
        // Receive data from the server and store it directly in the fileDataBuffer
        size_t totalBytesReceived = 0;
        while (totalBytesReceived < FILE_DATA_BUFFER_SIZE) {
            int bytesReadFromServer = recv(serverSock, fileDataBuffer + totalBytesReceived, FILE_DATA_BUFFER_SIZE - totalBytesReceived, 0);
            if (bytesReadFromServer <= 0) {
                // End of file or error receiving from server
                break;
            }
            totalBytesReceived += bytesReadFromServer;
        }
        
        printf("Received the full file from the server.\n");
        gettimeofday(&fetch_end, NULL);
        
        gettimeofday(&filter_start, NULL);
        size_t filtered_size;
        filteredBuffer = processParquetData(fileDataBuffer, totalBytesReceived, &filtered_size);
        gettimeofday(&filter_end, NULL);

        if (fileDataBuffer == NULL) {
            perror("Error filtering buffer");
            exit(1);
        }
    
        // Forward data from the filtered buffer to the client
        size_t bytesSentToClient = 0;
        while (bytesSentToClient < filtered_size) {
            int bytesSent = send(clientSock, filteredBuffer + bytesSentToClient, filtered_size - bytesSentToClient, 0);
            if (bytesSent <= 0) {
                perror("Error sending data to client");
                break;
            }
            bytesSentToClient += bytesSent;
        }
        
        printf("File forwarded between the client and the server.\n");
        	
        long long fetch_time_ms = get_elapsed_milliseconds(fetch_start, fetch_end);
        printf("File received in %lld milliseconds\n", fetch_time_ms);


	    long long filter_time_ms = get_elapsed_milliseconds(filter_start, filter_end);
        printf("Filtered in %lld milliseconds\n", filter_time_ms);

        
        // Clean up
        
        // Close the sockets
        close(clientSock);
        close(serverSock);
    }
    
    free(fileDataBuffer); // Free the allocated memory
    free(filteredBuffer);
    return 0;
}
