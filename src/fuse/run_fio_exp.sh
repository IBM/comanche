#/bin/bash

# you shall start the server first:
#  ./src/fuse/ustack/kv_ustack -d /tmp/kvfs-ustack


if [ "x"${NO_PRELOAD} == "x" ]; then
  export LD_PRELOAD=./src/fuse/ustack/libustack_client.so
fi

BS=1m SIZE=1m DIRECTORY=/tmp/kvfs-ustack fio ../src/fuse/kv-ustack.fio
