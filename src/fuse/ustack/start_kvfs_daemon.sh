#!/usr/bin/env bash
KB=1024
app="./src/fuse/ustack/kv_ustack"

# configuables
fio_options="-o max_write=131072 -o big_writes"
mount_dir="/tmp/kvfs-ustack"
export USTACK_PAGE_SIZE=$((4*KB))

function usage()
{
    echo "help info for kvfs-ustack daemon"
    echo ""
    printf "./$0"
    printf "\t-h| --help"
    printf "\t-d|--debug"
    printf "\t-p|--profile"
    printf "\t--mount-dir=${mount_dir}"
    printf "\t--pgsize=$USTACK_PAGE_SIZE(4096|65536|2097152) \n"
    echo ""
}

function start_kvfs_daemon {
  mount | grep -q "ustack"
  if [ $? == 0 ]; then
    printf "ERROR; shall unmount first with\n\t fusermount -u $mount_dir\n"
    exit
  fi
	printf "# starting kvfs-ustack deamon with:page size=$USTACK_PAGE_SIZE, mount_dir=${mount_dir}, fio_options=${fio_options}\n"
  $app ${mount_dir} $fio_options
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        -d | --debug)
  					fio_options=$fio_options" -d"
            ;;
        -p | --profile)
            app="${app}_with_profiler"
            ;;
        --pgsize)
						export USTACK_PAGE_SIZE=$VALUE
            ;;
        --mount-dir)
            if [ ! -d $VALUE ]; then
              echo "ERROR: mount dir $VALUE doesn't exists"
              exit
            fi
						export mount_dir=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

start_kvfs_daemon

