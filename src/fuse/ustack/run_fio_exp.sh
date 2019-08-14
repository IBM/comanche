#/bin/bash

# you shall start the server first:

# To use filestore instead of nvmestore
# export KVFS_BACKEND="filestore"
#  ./src/fuse/ustack/kv_ustack -d /tmp/kvfs-ustack

unset PRELOAD_CMD

if [ "x"${DIRECTORY} == "x" ]; then
  METHOD=kvfs
  export DIRECTORY=/tmp/kvfs-ustack
else
  METHOD=ext4
  NO_PRELOAD="1"
fi

if [ "x"${NO_PRELOAD} == "x" ]; then
  PRELOAD_CMD=./src/fuse/ustack/libustack_client.so
  METHOD=${METHOD}-ustack
fi

export FILESIZE=16m
CURDATE=`date '+%F'`
CURTIME=`date '+%F-%H-%M'`
RESULTDIR=results/fio/${CURDATE}
mkdir -p ${RESULTDIR}

export SYNC=1
KB=1024

# run 4k random write for latency percentile
run_latency_percentile_exp() {
# exp1 4k random writes
for exp in {1..1}; do
  echo "run exp ${exp}:"
  export DIRECT=1
  export BS=4k 
  local RESULTPATH=${RESULTDIR}/fio-${METHOD}-bs-${BS}-direct-${DIRECT}-sync-${SYNC}.json
  LD_PRELOAD=${PRELOAD_CMD} fio ../src/fuse/fio/rand4kwrite.fio --output-format=json+ 1> $RESULTPATH
  echo "Results saved in ${RESULTPATH}"
done
}

# increasing io size
# use the following cmd to print statistics:
# less results/fio/fio-kvfs-ustack-varied-iosizes-direct-1-sync-1.json|grep "\"write\"" -A 6|grep bw

run_randwrite_exp() {
export DIRECT=1

export USTACK_PAGE_SIZE=$((4*KB))
#export USTACK_PAGE_SIZE=$((64*KB))
#export USTACK_PAGE_SIZE=$((1024*KB))

RESULTPATH=${RESULTDIR}/fio-${METHOD}-varied-iosizes-pgsize-${USTACK_PAGE_SIZE}-direct-${DIRECT}-sync-${SYNC}.json
echo "Results Generated from $HOSTNAME at ${CURTIME}:" > $RESULTPATH
for exp in {1..1}; do
  echo "## run exp ${exp}:"
  for IOSIZE in 4k 16k 64k 256k 1024k 4096k; do
    echo "### run with IOSIZE ${IOSIZE}:"
    LD_PRELOAD=${PRELOAD_CMD} BS=${IOSIZE} fio ../src/fuse/fio/rand4kwrite.fio --output-format=json+ 1>>$RESULTPATH
  done
done

echo "Results saved in ${RESULTPATH}"
}

run_latency_percentile_exp
#run_randwrite_exp
