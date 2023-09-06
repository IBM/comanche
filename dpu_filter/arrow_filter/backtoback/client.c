#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>

#define TCP_PORT 8888
#define SERVER_IP "192.168.0.156" // Replace with the server's IP address
#define BUFFER_SIZE (1024*1024) //1024

long long get_elapsed_milliseconds(struct timeval start_time, struct timeval end_time) {
    return (end_time.tv_sec - start_time.tv_sec) * 1000LL +
           (end_time.tv_usec - start_time.tv_usec) / 1000LL;
}

int main() {
    int sockfd;
    struct sockaddr_in serverAddr;
    char buffer[BUFFER_SIZE];

    // Create a TCP socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Error in socket");
        exit(1);
    }

    memset(&serverAddr, 0, sizeof(serverAddr));

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(TCP_PORT);
    if (inet_aton(SERVER_IP, &serverAddr.sin_addr) == 0) {
        perror("Error in inet_aton");
        exit(1);
    }

    // Connect to the server
    if (connect(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Error in connect");
        exit(1);
    }

    printf("Connected to the server.\n");

    // Get the filename from the user
    printf("Enter the filename you want to request: ");
    scanf("%s", buffer);

    // Measure the time before sending the file name
    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    // Send the filename to the server
    int bytesSent = send(sockfd, buffer, strlen(buffer), 0);
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
        int bytesRead = recv(sockfd, buffer, BUFFER_SIZE, 0);
        if (bytesRead < 0) {
            // Error receiving
            perror("Error receiving data from server");
            break;
        } else if (bytesRead == 0) {
            // End of file
            break;
        }

        // Write data to the file
        int bytesWritten = fwrite(buffer, 1, bytesRead, file);
        if (bytesWritten < bytesRead) {
            perror("Error writing to file");
            break;
        }
    }

    // Measure the time after receiving the file
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    long long elapsed_time_ms = get_elapsed_milliseconds(start_time, end_time);
    printf("File received in %lld milliseconds\n", elapsed_time_ms);

    // Close the file and the socket
    fclose(file);
    close(sockfd);

    return 0;
}
