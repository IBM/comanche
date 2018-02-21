#include <common/exceptions.h>
#include <common/errors.h>
#include <core/dpdk.h>
#include <string.h>

#include "blob_component.h"

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

/* factory - late binding version */
Component::IBlob *
Blob_component_factory::
open(std::string owner,
     std::string name,
     int flags)
{
  auto obj = static_cast<Component::IBlob *>
    (new Blob_component(owner, name, nullptr, flags));

  obj->add_ref();
  return obj;
}


/* main component */

Blob_component::Blob_component(std::string owner,
                               std::string name,
                               Component::IBlock_device * base_block_device,
                               int flags)
  : _owner(owner),
    _name(name),
    _base_block_device(base_block_device),
    _flags(flags)
{
  if(_base_block_device) {
    _base_block_device->add_ref();
    _base_block_device->get_volume_info(_base_block_device_vi);
    
    if(_base_block_device_vi.block_size != BLOCK_SIZE)
      throw Constructor_exception("blob component requires 4K underlying block size; size is (%lu)",
                                  _base_block_device_vi.block_size);
    
  
    if(((METADATA_SIZE_BYTES / _base_block_device_vi.block_size) +
        (ALLOCATOR_SIZE_BYTES / _base_block_device_vi.block_size) + 1024) > _base_block_device_vi.block_count)
      throw Constructor_exception("device too small");
    
    instantiate_components();
  }
}

Blob_component::~Blob_component()
{
  if(_base_block_device) {
    /* release components */
    _metadata->release_ref();
    _allocator->release_ref();
    _pmem_allocator->release_ref();
    _block_data->release_ref();
    _block_allocator->release_ref();
    _block_md->release_ref();
    _base_rm->release_ref();
    _base_block_device->release_ref();
  }
}

int Blob_component::bind(Component::IBase * base)
{
  if(_base_block_device)
    throw API_exception("bind: base block device already bound");
  
  if(!base)
    throw API_exception("bind: base param is null");

  _base_block_device = (Component::IBlock_device *) base->query_interface(Component::IBlock_device::iid());

  if(!_base_block_device) {
    return 1;
  }
  else {
    instantiate_components();    
    return 0;
  }
}

void Blob_component::flush_md()
{
  assert(_block_md);
  //  _block_md->write(_block_md_iob,0,0,NUM_BLOCKS_FOR_METADATA); /*< writes out whole of slab */
}

struct Blob_handle
{
public:
  void * block_allocator_handle;
  lba_t  lba;
  size_t lba_count;
  Component::io_buffer_t iob;
  void * vaddr;
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
create(const std::string& name,
       const std::string& owner,
       const std::string& datatype,
       size_t size_in_bytes)
{
  if(!_base_block_device)
    throw API_exception("late binding still not bound");
  
  assert(size_in_bytes > 0);
  
  Blob::Data_region* dr;  
  size_t n_blocks = (size_in_bytes + _base_block_device_vi.block_size - 1)  / _base_block_device_vi.block_size;  

  Blob_handle* handle = new Blob_handle;
  handle->lba_count = n_blocks;

  /* allocate data block range */
  handle->lba = _allocator->alloc(n_blocks, &handle->block_allocator_handle);

  /* allocate metadata */
  _metadata->allocate(handle->lba, n_blocks, name, owner, datatype);

  /* allocate IO buffer */
  size_t rounded_size_in_bytes = n_blocks * _base_block_device_vi.block_size;
  handle->iob = _block_data->allocate_io_buffer(rounded_size_in_bytes,
                                               _base_block_device_vi.block_size,
                                               -1);
  handle->vaddr = _block_data->virt_addr(handle->iob);
  
  /* zero data in memory and on disk */
  memset(handle->vaddr, 0, rounded_size_in_bytes);
  _block_data->write(handle->iob, 0, handle->lba, handle->lba_count, 0);

  /* return handle */
  return static_cast<IBlob::blob_t>(handle);  
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
  if(!_base_block_device)
    throw API_exception("late binding still not bound");

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
  _metadata->dump_info();
  
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

bool Blob_component::check_key(const std::string& key, size_t& out_size)
{
  return _metadata->check_exists(key, "", out_size);
}

void Blob_component::get_metadata_vector(const std::string& filter,
                                         std::vector<std::string>& out_vector)
{
  IMetadata::iterator_t i = _metadata->open_iterator(filter);
  index_t index;
  std::string md;
  uint64_t lba, lba_count;
  while(_metadata->iterator_get(i, index, md, &lba, &lba_count) == S_OK) {
    out_vector.push_back(md);
  }
}

void
Blob_component::instantiate_components()
{
  PINF("Blob component flags=%d", _flags);
  
  /* region manager */
  {
    IBase * comp = load_component("libcomanche-partregion.so",
                                  Component::part_region_factory);
    assert(comp);
    
    IRegion_manager_factory* fact = (IRegion_manager_factory *) comp->query_interface(IRegion_manager_factory::iid());
    assert(fact);
    
    _base_rm = fact->open(_base_block_device, _flags);

    fact->release_ref();  
  }

  if(option_DEBUG) PLOG("Blob: region manager created");
  
  addr_t vaddr;
  bool reused;
  
  /* create block device for 'allocator' */
  std::string region_name = _name + "-allocator";
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
  region_name = _name + "-md";
  size_t md_nblocks = METADATA_SIZE_BYTES / _base_block_device_vi.block_size;
  _block_md = _base_rm->reuse_or_allocate_region(METADATA_SIZE_BYTES / _base_block_device_vi.block_size,
                                                 _owner, region_name, vaddr, reused);
  assert(_block_md);
  if(option_DEBUG) PLOG("Blob: block device for metadata created (%lu MB)",
                        REDUCE_MB(md_nblocks * _base_block_device_vi.block_size));
  
  /* create block device for 'data' */
  region_name = _name + "-data";
  size_t data_nblocks = _base_block_device_vi.block_count - md_nblocks - allocator_nblocks;
  _block_data = _base_rm->reuse_or_allocate_region(data_nblocks,
                                                   _owner, region_name, vaddr, reused);
  assert(_block_data);
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
                                      "block-alloc-ut",
                                      0, /* numa node */
                                      _flags);  
    fact->release_ref();
    assert(_allocator);
    if(option_DEBUG) PLOG("Blob: allocator created");
  }

  /* create metadata */
  {
    IBase * comp = load_component("libcomanche-mdfixobd.so",
                                  Component::metadata_fixobd_factory);

    IMetadata_factory * fact = static_cast<IMetadata_factory *>
      (comp->query_interface(IMetadata_factory::iid()));

    _metadata = fact->create(_block_md, _base_block_device_vi.block_size, _flags);
    assert(_metadata);

    fact->release_ref();
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

