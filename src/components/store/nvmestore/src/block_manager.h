/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef BLOCK_MANAGER_H_
#define BLOCK_MANAGER_H_

#include <api/block_allocator_itf.h>
#include <api/block_itf.h>
#include <string>
#include "common/types.h"  // status_t

using namespace Component;
namespace nvmestore
{
/* type of block io*/
enum {
  BLOCK_IO_NOP   = 0,
  BLOCK_IO_READ  = 1,
  BLOCK_IO_WRITE = 2,
};
class Block_manager {
  static constexpr size_t CHUNK_SIZE_IN_BLOCKS =
      8;  // large IO will be splited into CHUNKs, 8*4k  seems gives optimal

 private:
  static constexpr bool option_DEBUG = false;
  size_t                _blk_sz;
 public:
  using io_buffer_t = uint64_t;
  Block_manager()   = delete;

  Block_manager(const std::string &pci, const std::string &pm_path)
      : _pci_addr(pci), _pm_path(pm_path), _blk_sz(4096)
  {
    status_t ret;

    ret = open_block_device(_pci_addr, _blk_dev);
    Component::VOLUME_INFO info;
    _blk_dev->get_volume_info(info);
    _blk_sz = info.block_size;

    if (S_OK != ret) {
      throw General_exception("failed (%d) to open block device at pci %s\n",
                              ret, pci.c_str());
    }

    ret = open_block_allocator(_blk_dev, _blk_alloc);
    if (S_OK != ret) {
      throw General_exception(
          "failed (%d) to open block block allocator for device at pci %s\n",
          ret, pci.c_str());
    }

    PDBG("block_manager: using block device %p with allocator %p, blksize= %lu",
         _blk_dev, _blk_alloc, _blk_sz);
  };

  ~Block_manager()
  {
    assert(_blk_dev);
    assert(_blk_alloc);
    _blk_alloc->release_ref();
    _blk_dev->release_ref();
  }

  /*
   * Issue block device io to one block device
   *
   * @param block block device
   * @param type read/write
   * @param mem io memory
   * @param lba block address
   * @param nr_io_blocks block to be operated on, all the blocks should fit in
   * the IO memory
   *
   * This call itself is synchronous
   */
  status_t do_block_io(int         type,
                       io_buffer_t mem,
                       uint64_t    lba,
                       size_t      nr_io_blocks);

  size_t blk_sz() const { return _blk_sz; }

  /* Inline Memory related Methods*/
  inline void *   virt_addr(io_buffer_t buffer) { return _blk_dev->virt_addr(buffer); }

  inline io_buffer_t allocate_io_buffer(size_t size, unsigned alignment, int numa_node)
  {
    return _blk_dev->allocate_io_buffer(size, alignment, numa_node);
  };

  inline void free_io_buffer(io_buffer_t io_mem) { _blk_dev->free_io_buffer(io_mem); };

  inline io_buffer_t register_memory_for_io(void *vaddr, addr_t paddr, size_t len)
  {
    return _blk_dev->register_memory_for_io(vaddr, paddr, len);
  }

  /* Block Allocator */
  lba_t alloc_blk_region(size_t size, void **handle = nullptr)
  {
    return _blk_alloc->alloc(size, handle);
  }

  void free_blk_region(lba_t addr, void *handle = nullptr)
  {
    _blk_alloc->free(addr, handle);
  }

 private:
  Component::IBlock_device *   _blk_dev;
  Component::IBlock_allocator *_blk_alloc;

  std::string _pci_addr;
  std::string _pm_path;
  /*
   * open the block device, reuse if it exists already
   *
   * @param pci in pci address of the nvme
   *   The "pci address" is in Bus:Device.Function (BDF) form with Bus and
   * Device zero-padded to 2 digits each, e.g. 86:00.0 The domain is implicitly
   * 0000.
   * @param block out reference of block device
   *
   * @return S_OK if success
   */
  status_t open_block_device(const std::string &        pci,
                             Component::IBlock_device *&block);

  /*
   * open an allocator for block device, reuse if it exsits already
   */

  status_t open_block_allocator(Component::IBlock_device *    block,
                                Component::IBlock_allocator *&alloc);

};  // Block_manager
}  // namespace nvmestore
#endif
