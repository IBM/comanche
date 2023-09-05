#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define MIDDLE_IP "192.168.0.156"
#define MIDDLE_PORT 8888
#define FILENAME_SIZE 256

int main() {
    int sockfd;
    struct sockaddr_in middle_addr;
    char filename[FILENAME_SIZE];
    FILE *file;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Error creating socket");
        exit(EXIT_FAILURE);
    }

    memset(&middle_addr, 0, sizeof(middle_addr));
    middle_addr.sin_family = AF_INET;
    middle_addr.sin_addr.s_addr = inet_addr(MIDDLE_IP);
    middle_addr.sin_port = htons(MIDDLE_PORT);

    if (connect(sockfd, (struct sockaddr *)&middle_addr, sizeof(middle_addr)) < 0) {
        perror("Error connecting to middle machine");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("Enter file name: ");
    scanf("%s", filename);
    send(sockfd, filename, strlen(filename), 0);

    // Receive file from middle machine
    file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    char buffer[1024];
    int bytes_received;
    while ((bytes_received = recv(sockfd, buffer, sizeof(buffer), 0)) > 0) {
        fwrite(buffer, 1, bytes_received, file);
    }

    fclose(file);
    close(sockfd);
    return 0;
}
