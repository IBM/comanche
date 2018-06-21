A kvstore which has metadata management in AEP.
Designed for large blobs.

Two types of meta data are maintained:
1. A mapping between ( object name, blobs) (see kvstore.cpp)
2. A bookkeep of all the allocation blocks ( implemented in allocator-blk-aep)

Data(blob) is stored in :
1. block-nvme

Block allocator using AEP

* Use pmem transaction api to manage block allocation using AEP.
* The allocation information can be either a avl tree or bitmap. 
