#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

#define SERVER_IP "10.10.10.111"
#define SERVER_PORT 8888
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    char filename[BUFFER_SIZE];
    char buffer[BUFFER_SIZE];
    FILE *file;
    clock_t start_time, end_time;

    // Create UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Error creating socket");
        exit(EXIT_FAILURE);
    }

    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(SERVER_PORT);

    while (1) {
        printf("Enter file name: ");
        scanf("%s", filename);
        
        // Measure the time before sending the file name
        start_time = clock();
        // Send file name request to server
        sendto(sockfd, filename, strlen(filename), 0, (struct sockaddr *)&server_addr, sizeof(server_addr));

        // Create file to store received data
        file = fopen(filename, "wb");
        if (file == NULL) {
            perror("Error opening file");
            continue;
        }

        // Receive and write file data
        while (1) {
            memset(buffer, 0, sizeof(buffer));
            int bytes_received = recvfrom(sockfd, buffer, sizeof(buffer), 0, NULL, NULL);
            if (bytes_received <= 0) {
                break;
            }

            // Check for the END marker
            if (strcmp(buffer, "END") == 0) {
                break;
            }

            fwrite(buffer, 1, bytes_received, file);
        }

        fclose(file);
        // Measure the time after receiving the file
        end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("File received: %s, Time taken: %.2f seconds\n", filename, elapsed_time);

    }

    close(sockfd);
    return 0;
}
