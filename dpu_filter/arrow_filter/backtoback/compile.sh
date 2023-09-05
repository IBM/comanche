#!/bin/bash

gcc server.c -o server
gcc client.c -o client
gcc dpu_socket.c -o dpu_socket

echo "Compilation completed."
