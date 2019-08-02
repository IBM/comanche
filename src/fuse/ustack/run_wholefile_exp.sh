#!/usr/bin/env bash

BINARY="./src/fuse/ustack/unit_test/test-preload" 
MOUNT_PATH="/tmp/mount" 

for IO_SIZE_IN_KB in 4 16 64 256 1024 4096 16384 
do
  for EXP_NAME in {1..3} 
  do

    echo "########### Runing ${BINARY} with MOUNT_PATH=${MOUNT_PATH}, IO_SIZE=${IO_SIZE_IN_KB}KB, exp (${EXP_NAME}/3)"

  $BINARY $MOUNT_PATH $IO_SIZE_IN_KB 
  done
done
