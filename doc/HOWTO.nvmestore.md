Prepare
=================

Find setup nvme device
----------------------

1. Get your pcie address of the nvme:
```
lspci|grep Non
```

2. sudo ./tools/attach_to_vfio.sh 11:00.0
(there is also a attach_to_nvme.sh to deattach nvme device)


Environment checks
------------------

```
sudo ./tools/prepare_nvmestore.sh
```

The script does the following check to ensure nvmestore can run properly:
1. check kernel boot cmdline
2. check pmem configurations
3. check spdk related setup(vfio, hugepage)
4. check comanche-specific setup(xms and /dev/hugepages permissions)
5. etc..

If the environment is not set correctly the script will try to fix it. For other special configuration errors (e.g. those requires system reboot) the script will redirect you to more detailed info here.

Mostly you may want to use the script in those two occasions:
1. after a fresh installation of comanche, and before running the unit test
2. someone else has did some thing with vfio and changed the permissions, then you need to change it back.

Backends(METAstore)
-------------------

nvmestore can use hstore or filestore as metastore to save block allocation data and object metadata.
By default filestore will be used. 

When FileStore is used, ...... 
 
When hstore is used and there is no pmem available in the system, one can emulate it with(required for each shell which initiates nvmestore component):

```
export USE_DRAM=24; export NO_CLFLUSHOPT=1; export DAX_RESET=1.
```


Unit Test
==========

In build/src/components/store/nvmestore/unit_test/:
1. test-nvmestore-basic(basic functionality)
2. test-nvmestore-integrity(check data integrity of nvmestore using crc32 checksums)
3. test-nvmestore-throughput

Performance
============

Currently nvmestore supports put/get/get_direct in kvperf --test.

1. run the test
```
./testing/kvstore/kvstore-perf --component nvmestore --pci_addr 11:00.0 --test=put --value_length=1048576 --elements=1000
```
(if not specifying --test=put, all tests will run some might fail but results for put/get/get_direct will be generated)

2. Generate result plot
```
python testing/kvstore/plot_results/plot_everything_in_file.py ./build/results/nvmestore/results_2019_05_21_17_19.json
```

DAWN test
------------

**In Build directory**

1. Prepare the server
```
./src/servers/dawn/dawn --config ../src/components/store/nvmestore/nvmestore-dawn.conf
```

2. Test with simple dawn-client
```
./src/components/client/dawn/unit_test/dawn-client-test1 --server-addr=10.0.0.82:11911
```

3. Test with kvstorre-pef dawn client
```
./dist/bin/kvstore-perf --component dawn --cores 22 --test get --server 10.0.0.82 --port 11911 --device_name mlx
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
