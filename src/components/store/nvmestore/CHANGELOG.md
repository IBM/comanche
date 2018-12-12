# Changelog
This is the changelog for nvmestore
[2018-12-12]:
#notes:
1. register directly memory, take a look at the dawn
2. lock/unlock.
  * currently lock trigger allocate io memory, this need to be fixed
  * check lock when operating during puts/gets?, check unit\_test of other store(e.g. hstore)

[2018-11-12]:
#todo:
  - change free to free\_memory after nvmestore-get
  - use 'take\_lock' for the apply method
  - adjust the hardcopied sizes
  - memio management (DEFAULT_IO_MEM_SIZE)
  - test more cores
  - put\_direct


[2018-07-31]: 
#added:
  - more documents
  - test-nvmestore now can be also used to test file store performance(temprary use)
#todo:
  - support new kv interface(uint64_t hashkey)
  - Fuse docs/and perf
  - get() currently gives you the address of the iomem, which user cannot free. 
#finding:
  - filestore without file cache has poor performance

