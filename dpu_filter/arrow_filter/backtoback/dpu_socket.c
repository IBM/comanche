#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SERVER_IP "192.168.0.42"
#define SERVER_PORT 8888
#define MIDDLE_PORT 8888
#define FILENAME_SIZE 256

int main() {
    int middle_sock, server_sock, client_sock;
    struct sockaddr_in middle_addr, server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char filename[FILENAME_SIZE];
    FILE *file;

    middle_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (middle_sock < 0) {
        perror("Error creating middle socket");
        exit(EXIT_FAILURE);
    }

    memset(&middle_addr, 0, sizeof(middle_addr));
    middle_addr.sin_family = AF_INET;
    middle_addr.sin_addr.s_addr = INADDR_ANY;
    middle_addr.sin_port = htons(MIDDLE_PORT);

    if (bind(middle_sock, (struct sockaddr *)&middle_addr, sizeof(middle_addr)) < 0) {
        perror("Error binding middle socket");
        close(middle_sock);
        exit(EXIT_FAILURE);
    }

    if (listen(middle_sock, 1) < 0) {
        perror("Error listening on middle socket");
        close(middle_sock);
        exit(EXIT_FAILURE);
    }

    while (1) {
        client_sock = accept(middle_sock, (struct sockaddr *)&client_addr, &client_len);
        if (client_sock < 0) {
            perror("Error accepting client connection");
            continue;
        }

        // Receive file name from client
        memset(filename, 0, sizeof(filename));
        recv(client_sock, filename, sizeof(filename), 0);

        // Request file from server
        server_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (server_sock < 0) {
            perror("Error creating server socket");
            close(client_sock);
            continue;
        }

        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
        server_addr.sin_port = htons(SERVER_PORT);

        if (connect(server_sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            perror("Error connecting to server");
            close(server_sock);
            close(client_sock);
            continue;
        }

        send(server_sock, filename, strlen(filename), 0);

        // Send the file back to the client
        file = fopen(filename, "rb");
        if (file == NULL) {
            perror("Error opening file");
            close(server_sock);
            close(client_sock);
            continue;
        }

        char buffer[1024];
        int bytes_read;
        while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
            send(client_sock, buffer, bytes_read, 0);
        }

        fclose(file);
        close(server_sock);
        close(client_sock);
    }

    close(middle_sock);
    return 0;
}
