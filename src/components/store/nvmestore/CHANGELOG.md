# Changelog
This is the changelog for nvmestore

[2017-07-31]: 
#added:
  - more documents 
  - test-nvmestore now can be also used to test file store performance(temprary use)
#todo:
  - support new kv interface(uint64_t hashkey)
  - Fuse docs/and perf
  - get() currently gives you the address of the iomem, which user cannot free. 
#finding:
  - filestore without file cache has poor performance

