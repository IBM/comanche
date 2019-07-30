Prepare
==========

1. In terminal(build directory), start the fs service:
```
mkdir -p mount
fusermount -u mount
./src/fuse/ustack/kv_ustack -d ./mount/
```
(Dont ctrl-C to end the service, instead use the unmount command below)

2. Unmount
```
fusermount -u mount
```

Basic Test
==========
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
```
(base) fengggli@ribbit5(:):~/WorkSpace/comanche/build/src/fuse$fio ../../../src/fuse/kv-ustack.fio
random-write: (g=0): rw=randwrite, bs=(R) 32.0KiB-32.0KiB, (W) 32.0KiB-32.0KiB, (T) 32.0KiB-32.0KiB, ioengine=libaio, iodepth=4
random-read: (g=0): rw=randread, bs=(R) 32.0KiB-32.0KiB, (W) 32.0KiB-32.0KiB, (T) 32.0KiB-32.0KiB, ioengine=libaio, iodepth=4
fio-3.1
Starting 2 processes
random-write: Laying out IO file (1 file / 64MiB)
fio: native_fallocate call failed: Operation not supported
random-read: Laying out IO file (1 file / 64MiB)
fio: native_fallocate call failed: Operation not supported
fio: pid=0, err=38/file:filesetup.c:184, func=ftruncate, error=Function not implemented

random-write: (groupid=0, jobs=1): err= 0: pid=19073: Wed Jul 17 17:05:18 2019
  write: IOPS=4796, BW=150MiB/s (157MB/s)(64.0MiB/427msec)
    slat (usec): min=151, max=641, avg=206.20, stdev=39.58
    clat (nsec): min=1937, max=1241.5k, avg=624376.38, stdev=82689.73
     lat (usec): min=161, max=1890, avg=830.79, stdev=107.80
    clat percentiles (usec):
     |  1.00th=[  482],  5.00th=[  498], 10.00th=[  537], 20.00th=[  553],
     | 30.00th=[  594], 40.00th=[  603], 50.00th=[  611], 60.00th=[  652],
     | 70.00th=[  668], 80.00th=[  668], 90.00th=[  693], 95.00th=[  758],
     | 99.00th=[  906], 99.50th=[  979], 99.90th=[ 1156], 99.95th=[ 1156],
     | 99.99th=[ 1237]
  lat (usec)   : 2=0.05%, 250=0.05%, 500=5.13%, 750=89.11%, 1000=5.27%
  lat (msec)   : 2=0.39%
  cpu          : usr=5.63%, sys=0.00%, ctx=2050, majf=0, minf=10
  IO depths    : 1=0.1%, 2=0.1%, 4=99.9%, 8=0.0%, 16=0.0%, 32=0.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     issued rwt: total=0,2048,0, short=0,0,0, dropped=0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=4

Run status group 0 (all jobs):
  WRITE: bw=150MiB/s (157MB/s), 150MiB/s-150MiB/s (157MB/s-157MB/s), io=64.0MiB (67.1MB), run=427-427msec
```

Preload
============

This will overwrite:
1. malloc/free
2. open/close
3. read/write

```
D_PRELOAD=./src/fuse/ustack/libustack_client.so ./src/fuse/ustack/unit_test/test-preload
```
