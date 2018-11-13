/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Feng Li (fengggli@yahoo.com)
 *
 */

/*
 * helper functions to initialize/release block device and block allocators
 */
#include "nvme_store.h"

#include <gtest/gtest.h>
#include <mutex>

#include <common/utils.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <api/block_itf.h>
#include <api/block_allocator_itf.h>


#include <core/dpdk.h>

// don't change this, some functionanlity such as get_direct might only work with blk_nvme backend.
#define USE_SPDK_NVME_DEVICE

using namespace Component;

static std::unordered_map<std::string, IBlock_device *> _dev_map; //pci->blk
static std::unordered_map<IBlock_device *, IBlock_allocator *> _alloc_map; //blk->alloc

std::mutex _dev_map_mutex;
std::mutex _alloc_map_mutex;

status_t NVME_store:: open_block_device(std::string pci, IBlock_device* &block)
{
  std::lock_guard<std::mutex> guard(_dev_map_mutex);

  if(_dev_map.find(pci) == _dev_map.end()){
    PLOG("[%s]: creating new block device...", __func__);
    DPDK::eal_init(512);

#ifdef USE_SPDK_NVME_DEVICE
    Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                        Component::block_nvme_factory);

    assert(comp);
    IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
    
    cpu_mask_t cpus;
    cpus.add_core(2);

    block = fact->create(pci.c_str(), &cpus);
    assert(block);
    fact->release_ref();

#else
    Component::IBase * comp = Component::load_component("libcomanche-blkposix.so",
                                                        Component::block_posix_factory);
    assert(comp);
    PLOG("Block_device factory loaded OK.");

    IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
    std::string config_string;
    config_string = "{\"path\":\"";
    //  config_string += "/dev/nvme0n1";1
    config_string += "./blockfile.dat";
    //  config_string += "\"}";
    config_string += "\",\"size_in_blocks\":10000}";
    PLOG("config: %s", config_string.c_str());

    block = fact->create(config_string);
    assert(block);
    fact->release_ref();
#endif

    PINF("Block-layer component loaded OK (itf=%p)", block);
    _dev_map.insert(std::pair<std::string, IBlock_device*>(pci, block));
    return S_OK;
  }
  else{ 
    block = _dev_map[pci];
    block->add_ref();
    return S_OK;
  }
}

status_t NVME_store::open_block_allocator(IBlock_device *block,Component::IBlock_allocator* &alloc)
{

  std::lock_guard<std::mutex> guard(_alloc_map_mutex);

  if(_alloc_map.find(block) == _alloc_map.end()){
    PLOG("[%s]: creating new block allocator... ", __func__);
    using namespace Component;
    assert(block);
    size_t nr_blocks_tracked;  // actual blocks tracked by the allocator
    VOLUME_INFO devinfo;


    // TODO: remove hardcopied block size
    constexpr size_t TOO_MANY_BLOCKS = GB(128)/KB(4);

    IBase * comp = load_component("libcomanche-blkalloc-aep.so",
                                  Component::block_allocator_aep_factory);
    assert(comp);
    IBlock_allocator_factory * fact = static_cast<IBlock_allocator_factory *>
      (comp->query_interface(IBlock_allocator_factory::iid()));

    block->get_volume_info(devinfo);

    size_t nr_blocks = devinfo.block_count; // actual blocks from this device
    assert(nr_blocks);

    nr_blocks_tracked = nr_blocks> TOO_MANY_BLOCKS? TOO_MANY_BLOCKS:nr_blocks;

    PLOG("%s: Opening allocator to support %lu \/ %lu blocks", 
        __func__, nr_blocks_tracked, nr_blocks);

    persist_id_t id_alloc = std::string(devinfo.volume_name) + ".alloc.pool";

    alloc = fact->open_allocator(nr_blocks_tracked,
                                 PMEM_PATH_ALLOC,
                                 id_alloc);  
    fact->release_ref();  

    _alloc_map.insert(std::pair<IBlock_device *, IBlock_allocator *>(block, alloc));
    return S_OK;
  }
  else{
    alloc = _alloc_map[block];
    alloc->add_ref();
    return S_OK;
  }
}

status_t  NVME_store::do_block_io(Component::IBlock_device * block,
                         int type,
                         io_buffer_t mem,
                         lba_t lba,
                         size_t nr_io_blocks){
  switch (type){
    case BLOCK_IO_READ:

      if(nr_io_blocks < CHUNK_SIZE_IN_BLOCKS)
        block->read(mem, 0, lba, nr_io_blocks);
      else{
        uint64_t tag;
        lba_t offset = 0;  // offset in blocks

        // submit async IO
        do{
          tag = block->async_read(mem, offset*BLOCK_SIZE, lba+offset, CHUNK_SIZE_IN_BLOCKS);
          offset += CHUNK_SIZE_IN_BLOCKS;
        }while(offset < nr_io_blocks);

        // leftover
        if(offset > nr_io_blocks){
          offset -= CHUNK_SIZE_IN_BLOCKS;
          tag = block->async_read(mem, offset*BLOCK_SIZE, lba + offset, nr_io_blocks - offset);
        }
        while(!block->check_completion(tag)); /* we only have to check the last completion */
      }
      break;

    case BLOCK_IO_WRITE:
      if(nr_io_blocks < CHUNK_SIZE_IN_BLOCKS)
        block->write(mem, 0, lba, nr_io_blocks);
      else{
        uint64_t tag;
        lba_t offset = 0;  // offset in blocks

        // submit async IO
        do{
          tag = block->async_write(mem, offset*BLOCK_SIZE, lba+offset, CHUNK_SIZE_IN_BLOCKS);
          offset += CHUNK_SIZE_IN_BLOCKS;
        }while(offset < nr_io_blocks);

        // leftover
        if(offset > nr_io_blocks){
          offset -= CHUNK_SIZE_IN_BLOCKS;
          tag = block->async_write(mem, offset*BLOCK_SIZE, lba + offset, nr_io_blocks - offset);
        }
        while(!block->check_completion(tag)); /* we only have to check the last completion */
      }
      break;
    default:
      throw General_exception("not implemented");
      break;
  }
  return S_OK;
}
