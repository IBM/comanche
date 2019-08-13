#/bin/bash

# you shall start the server first:

# To use filestore instead of nvmestore
# export KVFS_BACKEND="filestore"
#  ./src/fuse/ustack/kv_ustack -d /tmp/kvfs-ustack

unset PRELOAD_CMD

if [ "x"${DIRECTORY} == "x" ]; then
  export METHOD=kvfs
  export DIRECTORY=/tmp/kvfs-ustack
else
  export METHOD=ext4
  export NO_PRELOAD="1"
fi

if [ "x"${NO_PRELOAD} == "x" ]; then
  export PRELOAD_CMD=./src/fuse/ustack/libustack_client.so
  export METHOD=${METHOD}-ustack
fi

export FILESIZE=16m
export RESULTDIR=results/fio
mkdir -p ${RESULTDIR}
# exp1 4k random writes
for exp in {1..1}; do
  echo "run exp ${exp}:"
  export SYNC=1
  export BS=4k 
  export RESULTPATH=${RESULTDIR}/fio-${METHOD}-bs-${BS}-sync-${SYNC}.json
  LD_PRELOAD=${PRELOAD_CMD} fio ../src/fuse/fio/rand4kwrite.fio --output-format=json+ 1> $RESULTPATH
  echo "Results saved in ${RESULTPATH}"
done
