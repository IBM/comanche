#include <common/exceptions.h>
#include <core/dpdk.h>
#include <string.h>

#include "blob_component.h"
#include "inode.h"

using namespace Component;

__attribute__((constructor))
static void __initialize_dpdk() {
  DPDK::eal_init(512, 1, true);
}


/* factory */
Component::IBlob *
Blob_component_factory::
open(std::string owner,
     std::string name,
     Component::IBlock_device * base_block_device,
     int flags)
{
  if(base_block_device == nullptr)
    throw API_exception("bad region interface parameter");

  auto obj = static_cast<Component::IBlob *>
    (new Blob_component(owner, name, base_block_device, flags));

  obj->add_ref();
  return obj;

}

/* main component */

Blob_component::Blob_component(std::string owner,
                               std::string name,
                               Component::IBlock_device * base_block_device,
                               int flags)
  : _owner(owner),
    _base_block_device(base_block_device)
{
  assert(_base_block_device);
  _base_block_device->add_ref();
  _base_block_device->get_volume_info(_base_block_device_vi);
  
  if(_base_block_device_vi.block_size != BLOCK_SIZE)
    throw Constructor_exception("blob component requires 4K underlying block size; size is (%lu)",
                                _base_block_device_vi.block_size);

  
  if(((MD_SIZE_BYTES / _base_block_device_vi.block_size) +
      (ALLOCATOR_SIZE_BYTES / _base_block_device_vi.block_size) + 1024) > _base_block_device_vi.max_lba)
    throw Constructor_exception("device too small");

  instantiate_components(flags & FLAGS_FORMAT, name);

  /* currently meta data is not a seperate component; later */
  _md = new Metadata(_block_md, flags & FLAGS_FORMAT);
   
}

Blob_component::~Blob_component()
{
  delete _md;

  /* release components */
  _allocator->release_ref();
  _pmem_allocator->release_ref();
  _block_data->release_ref();
  _block_allocator->release_ref();
  _block_md->release_ref();
  _base_rm->release_ref();
  _base_block_device->release_ref();
}

void Blob_component::flush_md()
{
  assert(_block_md);
  //  _block_md->write(_block_md_iob,0,0,NUM_BLOCKS_FOR_METADATA); /*< writes out whole of slab */
}

class Blob_handle
{
public:
  Blob_handle(Blob::Data_region * dr_param) : magic(0x0101), dr(dr_param)
  {
    assert(dr_param);
  }
  uint32_t magic;
  Blob::Data_region * dr;
};
  

/** 
 * Create a new blob
 * 
 * @param size_in_bytes Initial size of blob
 * 
 * @return Handle to new blob
 */
IBlob::blob_t
Blob_component::
create(size_t size_in_bytes)
{
  assert(size_in_bytes > 0);
  
  Blob::Data_region* dr;  
  size_t n_blocks = (size_in_bytes / BLOCK_SIZE) + 2; /* round-up + metadata */
  
  // {
  //   std::lock_guard<std::mutex> g(_md_lock);
  //   dr->set_as_head();
  //   flush_md();
  // }
  // PLOG("create: data region allocated %lu (%ld blocks)", dr->addr(), n_blocks);

  // /* construct and write out inode */
  // assert(dr->addr() % BLOCK_SIZE == 0);
  // addr_t hdr_lba = dr->addr() / BLOCK_SIZE;

  // Component::io_buffer_t iobuff = _block_data->allocate_io_buffer(BLOCK_SIZE, BLOCK_SIZE, NUMA_NODE_ANY);
  // struct inode_t * inode = static_cast<struct inode_t*>(_block_data->virt_addr(iobuff));
  // inode->magic = BLOB_INODE_MAGIC;
  // inode->flags = 0;
  // inode->nblocks = n_blocks;
  // inode->size = size_in_bytes;
  // inode->next_inode = 0;
  // memset(inode->ranges,0,sizeof(inode->ranges));
  // inode->ranges[0] = {dr->addr(), dr->addr() + n_blocks - 1};

  // _block_data->write(iobuff, 0, hdr_lba, 1);
  // _block_data->free_io_buffer(iobuff);

  // PLOG("created blob@ %lx size=%ld blocks", hdr_lba, n_blocks);
  // return reinterpret_cast<blob_t>(new Blob_handle(dr));
  return 0;
}

/** 
 * Erase a blob
 * 
 * @param handle Blob handle
 */
void
Blob_component::
erase(IBlob::blob_t handle)
{
  // Blob_handle * h = reinterpret_cast<Blob_handle*>(handle);
  // {
  //   std::lock_guard<std::mutex> g(_md_lock);
  //   _data_allocator->free(h->dr->addr()); /* TODO optimize since we have Memory_region */
  //   flush_md();
  // }
  // delete h;
}

/** 
 * Open a cursor to a blob
 * 
 * @param handle Blob handle
 * 
 * @return Cursor
 */
IBlob::cursor_t
Blob_component::open(IBlob::blob_t handle)
{
  return nullptr;
}

/** 
 * Close a cursor to a blob
 * 
 * @param handle 
 * 
 * @return 
 */
IBlob::cursor_t
Blob_component::
close(IBlob::blob_t handle)
{
  return nullptr;
}

/** 
 * Move cursor position
 * 
 * @param cursor Cursor handle
 * @param offset Offset in bytes
 * @param flags SEEK_SET etc.
 */
void
Blob_component::
seek(cursor_t cursor, long offset, int flags)
{
}

/** 
 * Zero copy version of read
 * 
 * @param cursor 
 * @param buffer 
 * @param buffer_offset 
 * @param size_in_bytes 
 */
void
Blob_component::
read(cursor_t cursor, io_buffer_t buffer, size_t buffer_offset, size_t size_in_bytes)
{
}

/** 
 * Copy-based read
 * 
 * @param cursor 
 * @param dest 
 * @param size_in_bytes 
 */
void
Blob_component::
read(cursor_t cursor, void * dest, size_t size_in_bytes)
{
}

/** 
 * Zero copy version of write
 * 
 * @param cursor 
 * @param buffer 
 * @param buffer_offset 
 * @param size_in_bytes 
 */
void
Blob_component::
write(cursor_t cursor, io_buffer_t buffer, size_t buffer_offset, size_t size_in_bytes)
{
}

/** 
 * Copy-based write
 * 
 * @param cursor 
 * @param dest 
 * @param size_in_bytes 
 */
void
Blob_component::
write(cursor_t cursor, void * dest, size_t size_in_bytes)
{
}


/** 
 * Set the size of the file (like POSIX truncate call)
 * 
 * @param size_in_bytes Size in bytes
 */
void
Blob_component::
truncate(blob_t handle, size_t size_in_bytes)
{
  // Blob_handle * h = reinterpret_cast<Blob_handle*>(handle);
  // if(size_in_bytes > h->dr->size())
  //   throw API_exception("truncation larger than existing size");

  // /* now we have to trim the storage blocks */
  // Inode inode(_block_data, h->dr->addr());

  // assert(inode.size() >= size_in_bytes);

  // if(inode.size() == size_in_bytes) return; /* no need to trim */

  // size_t new_nblocks = (size_in_bytes / BLOCK_SIZE) + 1; //bytes_to_trim = inode.size() - size_in_bytes;
  // size_t blocks_to_remove = inode.nblocks() - new_nblocks;

  // while(blocks_to_remove > 0) {
  //   range_t * r = inode.last_range();
  //   if(r==nullptr) throw Logic_exception("truncating blocks");

  //   size_t delta = (r->last - r->first);
  //   if(blocks_to_remove >= delta) {
  //     /* remove complete range */
  //     r->last = r->first = 0;
  //     blocks_to_remove -= delta;
  //   }
  //   else {
  //     /* shrink range */
  //     r->last -= blocks_to_remove;
  //     blocks_to_remove = 0;
  //   }
  // }

  // assert(size_in_bytes <= new_nblocks * BLOCK_SIZE);
  // /* update allocator */
  // {
  //   std::lock_guard<std::mutex> g(_md_lock);
  //   _data_allocator->trim(h->dr->addr(), new_nblocks * BLOCK_SIZE);
  // }

}

void
Blob_component::show_state(std::string filter)
{
  
  // if(filter.compare("*")==0) {
  //   PINF("-- BLOB STORE STATE --");
  //   _data_allocator->apply([](addr_t addr,size_t size, bool free, bool head)
  //                          {
  //                            if(free) return;
                             
  //                            if(head) {
  //                              PINF("Head: %08lx-%08lx",addr,addr+size);
  //                            }
  //                          });
  //   PINF("-- end ---------------");
  // }
}


void
Blob_component::instantiate_components(bool force_init, std::string& name)
{
  /* region manager */
  {
    IBase * comp = load_component("libcomanche-partregion.so",
                                  Component::part_region_factory);
    assert(comp);
    
    IRegion_manager_factory* fact = (IRegion_manager_factory *) comp->query_interface(IRegion_manager_factory::iid());
    assert(fact);
    
    _base_rm = fact->open(_base_block_device, force_init);

    fact->release_ref();  
  }

  if(option_DEBUG) PLOG("Blob: region manager created");
  
  addr_t vaddr;
  bool reused;
  
  /* create block device for 'allocator' */
  std::string region_name = name + "-allocator";
  assert(ALLOCATOR_SIZE_BYTES % _base_block_device_vi.block_size == 0);
  size_t allocator_nblocks = ALLOCATOR_SIZE_BYTES / _base_block_device_vi.block_size;
  _block_allocator = _base_rm->reuse_or_allocate_region(allocator_nblocks,
                                                        _owner, region_name, vaddr, reused);
  assert(_block_allocator);

  if(option_DEBUG) PLOG("Blob: block device for allocator created");

  {
    /* persistent memory for allocator */
    IBase * comp = load_component("libcomanche-pmemfixed.so",
                                  Component::pmem_fixed_factory);
    assert(comp);
    IPersistent_memory_factory * fact = static_cast<IPersistent_memory_factory *>
      (comp->query_interface(IPersistent_memory_factory::iid()));
    assert(fact);
    _pmem_allocator = fact->open_allocator(_owner, _block_allocator);
    assert(_pmem_allocator);
    fact->release_ref();
    _pmem_allocator->start();
    if(option_DEBUG) PLOG("Blob: pmem for allocator created");
  }
  
  
  /* create block device for 'metadata' */
  region_name = name + "-md";
  size_t md_nblocks = MD_SIZE_BYTES / _base_block_device_vi.block_size;
  _block_md = _base_rm->reuse_or_allocate_region(MD_SIZE_BYTES / _base_block_device_vi.block_size,
                                                 _owner, region_name, vaddr, reused);
  assert(_block_md);
  if(option_DEBUG) PLOG("Blob: block device for metadata created (%lu MB)",
                        REDUCE_MB(md_nblocks * _base_block_device_vi.block_size));
  
  /* create block device for 'data' */
  region_name = name + "-data";
  size_t data_nblocks = _base_block_device_vi.max_lba - md_nblocks - allocator_nblocks;
  _block_data = _base_rm->reuse_or_allocate_region(data_nblocks,
                                                   _owner, region_name, vaddr, reused);
  assert(_block_md);
  if(option_DEBUG) PLOG("Blob: block device for data created");

  if(option_DEBUG) 
    PLOG("Blob: %lu data blocks (%lu MB)",
         data_nblocks,
         REDUCE_MB(data_nblocks * _base_block_device_vi.block_size));

  /* create block allocator */
  {
    IBase * comp = load_component("libcomanche-allocblock.so",
                                  Component::block_allocator_factory);
    assert(comp);
    IBlock_allocator_factory * fact = static_cast<IBlock_allocator_factory *>
      (comp->query_interface(IBlock_allocator_factory::iid()));

    PLOG("Opening allocator to support %lu blocks", data_nblocks);
    _allocator = fact->open_allocator(_pmem_allocator,
                                      data_nblocks,
                                      "block-alloc-ut");  
    fact->release_ref();
    assert(_allocator);
    if(option_DEBUG) PLOG("Blob: allocator created");
  }

}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Blob_component_factory::component_id()) {
    return static_cast<void*>(new Blob_component_factory());
  }
  else
    return NULL;
}

