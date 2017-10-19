#ifndef __BLOB_INODE_H__
#define __BLOB_INODE_H__

#include <stdint.h>
#include <sys/types.h>
#include <common/types.h>
#include <api/memory_itf.h>

static constexpr uint16_t BLOB_INODE_MAGIC = 0xb10b;
static constexpr uint16_t BLOB_INODE_FLAG_LOCKED = 0x1;
static constexpr unsigned BLOB_INODE_RANGE_DIM = 254;

class Inode;


struct range_t {
  addr_t first;
  addr_t last;
} __attribute__((packed));

struct inode_t {
  uint16_t       magic;
  uint16_t       flags;
  size_t         size;
  uint64_t       resvd;
  uint32_t       nblocks;
  addr_t         next_inode;
  struct range_t ranges[BLOB_INODE_RANGE_DIM];
} __attribute__((packed));

class Cursor {
public:
  Cursor(Inode * inode) : _inode(inode) {
  }

  virtual ~Cursor() {
  }
  
private:
  Inode *                _inode;
  Component::io_buffer_t _iob;
  byte *                 _data;
};

/** 
 * Helper class for opening inodes
 * 
 * @param blk Block device
 * @param lba LBA of inode
 */
class Inode
{
public:
  static const unsigned BLOCK_SIZE = 4096;
  
public:
  Inode(Component::IBlock_device * blk, addr_t lba) :
    _blk(blk), _lba(lba) {
    assert(blk);
    _iob = blk->allocate_io_buffer(BLOCK_SIZE,BLOCK_SIZE,Component::NUMA_NODE_ANY);
    _inode = (inode_t *) _blk->virt_addr(_iob);
    _blk->read(_iob,0,lba,1);

    //    if(_inode->flags & BLOB_INODE_FLAG_LOCKED)
    //      throw Constructor_exception("inode locked");

    if(_inode->magic != BLOB_INODE_MAGIC)
      throw Constructor_exception("inode corrupt");

    lock_inode();

    dump_info();
  };
  
  virtual ~Inode() {
    unlock_inode();
    _blk->free_io_buffer(_iob);
  }

  void dump_info()
  {
    printf("INODE: magic=%04x\n", _inode->magic);
    printf("     : flags=%04x\n", _inode->flags);
    printf("     : nblocks=%d\n", _inode->nblocks);
    printf("     : next_inode=%lx\n", _inode->next_inode);
    printf("     : ");
    //    for(unsigned i=0;i<BLOB_INODE_RANGE_DIM;i++) {
    for(unsigned i=0;i<BLOB_INODE_RANGE_DIM;i++) {
      if(_inode->ranges[i].last > 0) {
        printf("[%u] %ld-%ld ",i, _inode->ranges[i].first,_inode->ranges[i].last);
      }
    }
    printf("\n");

  }

  range_t * last_range() const {
    int i = BLOB_INODE_RANGE_DIM-1;
    while(i >= 0) {
      if(_inode->ranges[i].last > 0) {
        return &_inode->ranges[i];
      }
      i--;
    }
    PNOTICE("last range has nothing i=%d",i);
    return nullptr;
  }
  
  void lock_inode()
  {
    _inode->flags |= BLOB_INODE_FLAG_LOCKED;
    _blk->write(_iob,0,_lba,1);    
  }
  
  void unlock_inode()
  {
    _inode->flags &= ~(BLOB_INODE_FLAG_LOCKED);
    _blk->write(_iob,0,_lba,1);
  }

  void seek(off64_t offset, int whence, addr_t* lba, unsigned* lba_offset)
  {
    
  }

  /* helpers */
  inline size_t size() const { return _inode->size; }
  inline size_t nblocks() const { return _inode->nblocks; }  


  
private:
  
  addr_t                     _lba;
  Component::IBlock_device * _blk;
  Component::io_buffer_t     _iob;
  inode_t *                  _inode;
};


static_assert(sizeof(inode_t) == Inode::BLOCK_SIZE,"invalid inode_t size");

#endif // __BLOB_INODE_H__
