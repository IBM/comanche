#/bin/bash

# you shall start the server first:

# To use filestore instead of nvmestore
# export KVFS_BACKEND="filestore"
#  ./src/fuse/ustack/kv_ustack -d /tmp/kvfs-ustack

unset PRELOAD_CMD

KB=1024

#default opts
METHOD=kvfs-ustack
mount_dir_ustack=/tmp/kvfs-ustack
mount_dir_ext4=/tmp/ext4-mount
export FILESIZE=16m

export SYNC=1
export DIRECT=1

function usage()
{
    echo "help info for kvfs-ustack daemon"
    echo ""
    printf "./$0\n"
    printf "\t-h| --help\n"
    printf "\t--method=$METHOD(kvfs-ustack|kvfs-naive|ext4) \n"
    printf "\t--pgsize=$USTACK_PAGE_SIZE(4096|65536|2097152, the pgsize set in daemon) \n"
    printf "\t--o_direct=${DIERCT}(0|1)\n"
    echo ""
}


CURDATE=`date '+%F'`
CURTIME=`date '+%F-%H-%M'`
RESULTDIR=results/fio/${CURDATE}
mkdir -p ${RESULTDIR}


# run 4k random write for latency percentile
run_latency_percentile_exp() {
# exp1 4k random writes
for exp in {1..1}; do
  echo "run exp ${exp}:"
  export BS=4k 
  local RESULTPATH=${RESULTDIR}/fio-${METHOD}-bs-${BS}-direct-${DIRECT}-sync-${SYNC}.json
  LD_PRELOAD=${PRELOAD_CMD} fio ../src/fuse/fio/rand4kwrite.fio --output-format=json+ 1> $RESULTPATH
  echo "Results saved in ${RESULTPATH}"
done
}

# increasing io size
# use the following cmd to print statistics (in kb):
# less results/fio/fio-kvfs-ustack-varied-iosizes-direct-1-sync-1.json|grep "\"write\"" -A 6|grep bw

run_randwrite_exp() {
if [ "x"$USTACK_PAGE_SIZE == "x" ] && [ $METHOD != "ext4" ]; then
  echo "ERROR: ustack page size not specified"
  exit
fi
#export USTACK_PAGE_SIZE=$((4*KB))
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

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;

        --pgsize)
						export USTACK_PAGE_SIZE=$VALUE
            ;;
        --method)
            if [ $VALUE == "kvfs-ustack" ]; then
              PRELOAD_CMD=./src/fuse/ustack/libustack_client.so
              export DIRECTORY=${mount_dir_ustack}

            elif [ $VALUE == "kvfs-naive" ]; then
              unset PRELOAD_CMD
              export DIRECTORY=${mount_dir_ustack}

            elif [ $VALUE == "ext4" ]; then
              mount|grep ${mount_dir_ext4} |grep -q nvme
              if [ $? != 0 ]; then
                echo "ERROR: path $mount_dir_ext4 not mounted on nvme"
                exit
              fi
              unset PRELOAD_CMD
              export DIRECTORY=${mount_dir_ext4}
            else
              echo "method $VALUE not supported"
              usage
              exit
            fi
            METHOD=$VALUE
            ;;
        --o_direct)
						export DIRECT=${VALUE}
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done


#run_latency_percentile_exp
run_randwrite_exp
