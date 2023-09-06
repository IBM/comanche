#!/bin/bash

gcc server.c -o server
gcc client.c -o client
gcc original_dpu.c -o original
g++ -c process_buffer.cc -o process_buffer.o -I/usr/local/include/arrow -L/usr/local/lib -lparquet -larrow -larrow_dataset
gcc -c dpu_socket.c -o dpu_socket.o -I/usr/local/include/arrow -L/usr/local/lib -lparquet -larrow -larrow_dataset
g++ -o dpu_socket dpu_socket.o process_buffer.o -lparquet -larrow -larrow_dataset
echo "Compilation completed."
