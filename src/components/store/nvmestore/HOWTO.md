Setup
-----------------
Run the [prepare_nvmestore.sh](comanche/tools/prepare_nvmestore.sh)
(Run with regular user, input sudo password when necessary)

If you met any issues, see [this REAME](comanche/src/components/store/nvmestore/README.md) for more information.

Run
------------------

Get your pcie address of the nvme:
```
lspci|grep Non
```

test-nvmestore
===============

This will check basic functionality:
```
PMEM_IS_PMEM_FORCE=1 ./comanche/src/components/store/nvmestore/unit_test/test-nvmestore pci-address 
```

Actually it can be also used to test the latency/throughput of file store, if you set the the USE_FILESTORE in the test-nvmestore.cpp

test-integrity
===============

TODO: I should merge two tests together..

This check data integrity of nvmestore using crc32 checksums
```
PMEM_IS_PMEM_FORCE=1 ./comanche/src/components/store/nvmestore/unit_test/test-integrity pci-address 
```

