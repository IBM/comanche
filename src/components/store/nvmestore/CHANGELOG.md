# Changelog
This is the changelog for nvmestore
[2019-05-29]:
#added:
* add value string in the persistent type, so that map_key can get the key string back
##TODO:
handle(BlockAllocRecord handle needs to be persist)
[2019-05-28]:
#added:
* nvmestore::map and its test
#wip:
* map_keys, I need std::string for key, which is not available in my hashtable, adding a new field?
* more tests

[2019-05-14]:
#fixed:
1. improve scripts.
2. Adapting to new create(debug_level, params) signature.
#todo:
1. Currently one allocator per blocknvme, it supports multiple stores above it. Using device name to identify each allocator can be revisited. 
2. delete_pool, attributes, mapapi.
3. use fs to keep metadata.
4. better management of pmem space.

[2018-12-13]:
#notes:
1. there is only one piece of iomem, instead I should use a slab allocator (similar to blkmeta)
2. How about blksize=512

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
