Changelog
==========
[19-08-15]:
  * Improved the partial write and fsync. 
    - Use pwrite(fio with ioengine=psync), instead of write.
    - I had to use ikvstore->debug api to sync an locked nvme object(instead of unlock followed by lock)
    - O_SYNC will be obtained in the fuse_info, and kvfs daemon will decide whether sync or not.
    - Currently dirty pages references are saved in std::set for simplicity.
    - Note: tradeoff in page cache size: smaller size cannot utilize nvme bandwidth, large size has write amplification.
  * scripts to start kvfs daemon(start_kvfs_daemon.sh) and fio workloads(run_fio_exp.sh)
  * results reporting(4k latency plot and varied-io-sizes throughput plot)
  * Add profiling, use -p when start the daemon

[19-08-09]:
  * Improved the kvfs-test unit test.
  * a different design, daemon side will lock fixed-size objects(default=2M) for each file during file open. Unlock will be called during file flush(which is called implicitedly by file close.). Fixed-sized objects serve as page cache in userspace.
  * Also support other ikvstore as backend, e.g. filestore.
  * wip on fio test.

[19-07-30]: Added 
  * 1-1 kvfs-ustack, where each file is mapped to one key-value pair.
  * First version only support wholefile operation without fileoffset.

Prepare
==========

1. In terminal(build directory), start the kvfs daemon:
```
MOUNTDIR=/tmp/kvfs-ustack
mkdir -p ${MOUNTDIR}

# you might need to unmount previous mount before doing this.
../src/fuse/ustack/start_kvfs_daemon.sh -d

(start_kvfs_daemon -h for more options), one important option is --pgsize, which changes the page cache size in the daemon.

(Dont ctrl-C to end the service, instead use the unmount command below)

2. Unmount
```
fusermount -u mount-dir
```

Basic Test of naive mode
========================

```
(base) fengggli@ribbit5(:):~/WorkSpace/comanche/build$echo "filecontent1" > mount/tmp.txt
$echo "filecontent2" > mount/tmp2.txt
$ls mount
tmp2.txt  tmp.txt
$cat mount/tmp.txt
filecontent1
$cat mount/tmp2.txt
filecontent2
```

FIO Test
=========

There are two experiments:
0. 4k persist writes in single file( to get latency percentile)
1. partial random persist write in a single file(to get throughput of different io sizes)

--pgsize is the pagesize set for daemon, this only used to generated the path name for the results..
```
../src/fuse/ustack/run_fio_exp.sh --method=kvfs-ustack --pgsize=4096 --exp=0
```
Results will be saved in a json file, whose path will be printed when job is done

Notes:

* run script with -h to see more options
* For exp.1, less xxx.json | grep "\"write\"" -A 6|grep bw, to see the bw(in KiB/s)

Preload(the client)
======================

This will overwrite:
1. malloc/free
2. open/close
3. read/write

```
LD_PRELOAD=./src/fuse/ustack/libustack_client.so ./src/fuse/ustack/unit_test/test-preload
```

Notes
=============

Profiling the daemon
--------------------
start the daemon with -p, will turn on profiler.

(with x-forwarding enabled):
google-pprof --gv src/fuse/ustack/kv_ustack cpu.profile


Mount with Linux ext4
------------------------

```
sudo ../tools/attach_to_nvme.sh  20:00.0
sudo mount  /dev/nvme0n1  /tmp/ext4-mount/
sudo chown -R lifen /tmp/ext4-mount/
```

Mount to comanche stack
```
sudo umount /tmp/ext4-mount
sudo ../tools/attach_to_vfio.sh  20:00.0

```

(test with DIRECTORY=/tmp/ext4-mount/ ../src/fuse/ustack/run_fio_exp.sh)
