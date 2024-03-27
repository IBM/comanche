#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h> // Include the time.h library for time measurement

#define TCP_PORT 12345
#define SERVER_IP "10.10.10.18" // Replace with the server's IP address
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in serverAddr;
    char buffer[BUFFER_SIZE];
    clock_t start_time, end_time; // Variables for measuring time

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

    // Record the start time
    start_time = clock();

    // Receive file contents from the server
    while (1) {
        // Receive data from the server
        int bytesRead = recv(sockfd, buffer, BUFFER_SIZE, 0);
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

    // Record the end time
    end_time = clock();

    // Calculate and print the elapsed time in milliseconds
    double elapsed_time = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;
    printf("File received from the server in %.2f milliseconds.\n", elapsed_time);

    // Close the file and the socket
    fclose(file);
    close(sockfd);

    return 0;
}