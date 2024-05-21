#!/bin/bash

gcc server.c -o server
gcc client.c -o client
gcc memclient.c -o memclient
gcc memserver.c -o memserver

echo "Compilation completed."

