Kernel parameters
-----------------

Use the following parameter, huge page is for pmdk, memmap is for the pmem, intel_iommu must be on for vfio
(if you are working in a machine with small memory size(e.g.no more than 8G), you should modify this correspondingly)

``` 
hugepagesz=2M hugepages=4096 intel_iommu=on text memmap=2G!4G
```

```
mkdir /mnt/huge
mount -t hugetlbfs nodev /mnt/huge

```

rerun tool/nvmesetup.sh

Enable the fake pmem
--------------------
```
./setup_pmem.sh
```


set pin memory limit for this user
-------------------------------------

if you run deps/dpdk/user-tools/dpdk-setup.sh to set vfio permission and get  warning to set the ulimit:
```
Current user memlock limit: 0 MB
```

Add those two lines to /et/security/limits.conf:
```  
  fengggli hard memlock unlimited 
  fengggli soft memlock unlimited
```

Note
---------------

you can se the clear_pmempool.sh  to clear all data from pmem

