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

#include <common/utils.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <api/block_itf.h>
#include <api/block_allocator_itf.h>


#include <core/dpdk.h>

#define USE_SPDK_NVME_DEVICE


using namespace Component;

void NVME_store:: init_block_device(std::string pci)
  {
    IBlock_device *block;

    DPDK::eal_init(512);

#ifdef USE_SPDK_NVME_DEVICE
    
    Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                        Component::block_nvme_factory);

    assert(comp);
    PLOG("Block_device factory loaded OK.");
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
    _blk_dev = block;
  }

  void NVME_store::init_block_allocator()
  {
    using namespace Component;
    assert(_blk_dev);
    size_t nr_blocks_tracked;  // actual blocks tracked by the allocator
    VOLUME_INFO devinfo;

    constexpr size_t TO_MANY_BLOCKS = GB(128)/KB(4);

    IBase * comp = load_component("libcomanche-blkalloc-aep.so",
                                  Component::block_allocator_aep_factory);
    assert(comp);
    IBlock_allocator_factory * fact = static_cast<IBlock_allocator_factory *>
      (comp->query_interface(IBlock_allocator_factory::iid()));

    _blk_dev->get_volume_info(devinfo);

    size_t nr_blocks = devinfo.block_count; // actual blocks from this device
    assert(nr_blocks);

    nr_blocks_tracked = nr_blocks> TO_MANY_BLOCKS? TO_MANY_BLOCKS:nr_blocks;

    PLOG("%s: Opening allocator to support %lu \/ %lu blocks", 
        __func__, nr_blocks_tracked, nr_blocks);

    persist_id_t id_alloc = std::string(devinfo.volume_name) + ".alloc.pool";

    _blk_alloc = fact->open_allocator(
                                  nr_blocks_tracked,
                                  PMEM_PATH_ALLOC,
                                  id_alloc);  
    fact->release_ref();  
  }
