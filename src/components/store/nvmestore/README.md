Setup
=================

Run the [prepare_nvmestore.sh][comanche/tools/prepare_nvmestore.sh] with sudo

The script does the following check to ensure nvmestore can run properly:
1. check kernel boot cmdline
2. check pmem configurations
3. check spdk related setup(vfi, hugepage)
4. check comanche-specific setup(xms and /dev/hugepages permissions)
5. etc..

If the environment is not set correctly the script will try to fix it. For other special configuration errors (e.g. those requires system reboot) the script will redirect you to more detailed info here.

Mostly you may want to use the script in those two occasions:
1. after a fresh installation of comanche, and before running the unit test
2. someone else has did some thing with vfio and changed the permissions, then you need to change it back.

Test and performance
=====================

unit test
--------------

This will check basic functionality:
1. test-nvmestore.
2. test-integrity
3. test-throughput

Performance
-----------

Currently nvmestore supports put/get/get_direct.

1. run the test
```
PMEM_IS_PMEM_FORCE=1 ./testing/kvstore/kvstore-perf --component nvmestore --pci_addr 11:00.0 --test=put --value_length=1048576 --elements=1000
```
(if not specify --test=put, all tests will run some might fail but results for put/get/get_direct will be generated)

2. Generate result plot
```
python testing/kvstore/plot_results/plot_everything_in_file.py ./build/results/nvmestore/results_2019_05_21_17_19.json
```

Other Information
==================

Kernel parameters
-----------------

Use the following parameter, huge page is for pmdk, memmap is for the pmem, intel_iommu must be on for vfio (tested with ubuntu 18.04 with kernel 4.15.0.20, no sure about higher kernel version)
(if you are working in a machine with small memory size(e.g.no more than 8G), you should modify this correspondingly)

* fill the following into GRUB_CMDLINE_LINUX in /etc/default/grub
``` 
intel_iommu=on hugepagesz=2M hugepages=4096 text memmap=2G!4G
```

* update grub
```
sudo update-grub
```

```
mkdir /mnt/huge
mount -t hugetlbfs nodev /mnt/huge

```

pmem setup
----------

```
./setup_pmem.sh
```

bypass clflush/msync
```
PMEM_IS_PMEM_FORCE=1 ./you-program
```

Memlock limit
---------------

if you run deps/dpdk/user-tools/dpdk-setup.sh to set vfio permission and get  warning to set the ulimit:
```
Current user memlock limit: 0 MB
```

Add those two lines to /et/security/limits.conf(this allows users in the sudo group allocate more io meory. ):
```  
  @sudo hard memlock unlimited 
  @sudo soft memlock unlimited
```

Remove Metadata
-----------------

you can use this to remove all metadata
```
rm -rf /mnt/pmem0/* 
```
