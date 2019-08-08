#/bin/bash

# you shall start the server first:
#  ./src/fuse/ustack/kv_ustack -d /tmp/kvfs-ustack


if [ "x"${NO_PRELOAD} == "x" ]; then
  export LD_PRELOAD=./src/fuse/ustack/libustack_client.so
fi

BS=4m SIZE=4m DIRECTORY=/tmp/kvfs-ustack SYNC=1 fio ../src/fuse/kv-ustack.fio
