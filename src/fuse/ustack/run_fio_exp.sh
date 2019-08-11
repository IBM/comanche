#/bin/bash

# you shall start the server first:

# To use filestore instead of nvmestore
# export KVFS_BACKEND="filestore"
#  ./src/fuse/ustack/kv_ustack -d /tmp/kvfs-ustack


if [ "x"${NO_PRELOAD} == "x" ]; then
  export LD_PRELOAD=./src/fuse/ustack/libustack_client.so
fi


BS=4k SIZE=32m DIRECTORY=/tmp/kvfs-ustack SYNC=1 fio ../src/fuse/kv-ustack.fio
