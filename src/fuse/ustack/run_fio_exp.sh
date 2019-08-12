#/bin/bash

# you shall start the server first:

# To use filestore instead of nvmestore
# export KVFS_BACKEND="filestore"
#  ./src/fuse/ustack/kv_ustack -d /tmp/kvfs-ustack

if [ "x"${DIRECTORY} == "x" ]; then
  export DIRECTORY=/tmp/kvfs-ustack
else
  export NO_PRELOAD="1"
fi

if [ "x"${NO_PRELOAD} == "x" ]; then
  export LD_PRELOAD=./src/fuse/ustack/libustack_client.so
fi



# exp1 4k random writes
BS=4k SIZE=16m SYNC=1 fio ../src/fuse/kv-ustack.fio
