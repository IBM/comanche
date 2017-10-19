#include <common/exceptions.h>
#include <string.h>
#include "bb_component.h"
#include "inode.h"

using namespace Component;


/* factory */
Component::IBlob *
Blob_component_factory::
open(std::string owner,
     std::string name,
     Component::IRegion_manager * region_device,
     size_t value_space_size_in_bytes,
     int flags)
{
  if(region_device == nullptr)
    throw API_exception("bad region interface parameter");

  auto obj = static_cast<Component::IBlob *>
    (new Blob_component(owner, name, region_device, value_space_size_in_bytes, flags));

  obj->add_ref();
  return obj;

}

/* main component */

Blob_component::Blob_component(std::string owner,
                               std::string name,
                               Component::IRegion_manager * region_device,
                               size_t value_space_size_in_bytes,
                               int flags)
  : _rm(region_device)
{

  PINF("Blob_component::ctor(flags=%d)",flags);
  _rm->add_ref();

  if(_rm->block_size() != BLOCK_SIZE)
    throw Constructor_exception("blob component requires 4K underlying block size");

  {
    Component::VOLUME_INFO vi;
    _rm->get_underlying_volume_info(vi);
    if(value_space_size_in_bytes > vi.max_lba * vi.block_size)
      throw Constructor_exception("insufficient space on device for requested blob store");    
  }
  
  /* open or create metadata region */
#if 0
  /* create value area */
  bool va_reused;
  std::string region_name = name + "-val";
  _block_data = _rm->reuse_or_allocate_region(value_space_size_in_bytes / _rm->block_size(),
                                              owner, region_name, &va_reused);
  assert(_block_data);
  _block_data->get_volume_info(_block_data_vi);

  PLOG("Data range allocator (0-%ld LBA)..reused=%d", _block_data_vi.max_lba,va_reused);

  /* create value range allocator */
  if(option_DEBUG) PLOG("Creating range allocator.");
  
  _data_allocator = new Blob::Data_region_tree<> (*_slab,0,_block_data_vi.max_lba * BLOCK_SIZE);
  if(option_DEBUG) _data_allocator->dump_info();
#endif
}

Blob_component::~Blob_component()
{
  delete _data_allocator;
  delete _slab;
  
  _block_md->free_io_buffer(_block_md_iob);
  _block_md->release_ref();
  _rm->release_ref();
}

void Blob_component::flush_md()
{
  assert(_block_md);
  _block_md->write(_block_md_iob,0,0,NUM_BLOCKS_FOR_METADATA); /*< writes out whole of slab */
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
  
  {
    std::lock_guard<std::mutex> g(_md_lock);
    dr  = _data_allocator->alloc(n_blocks * BLOCK_SIZE);
    if(!dr) throw General_exception("no space left to create blob (size=%ld)", size_in_bytes);
    dr->set_as_head();
    flush_md();
  }
  PLOG("create: data region allocated %lu (%ld blocks)", dr->addr(), n_blocks);

  /* construct and write out inode */
  assert(dr->addr() % BLOCK_SIZE == 0);
  addr_t hdr_lba = dr->addr() / BLOCK_SIZE;

  Component::io_buffer_t iobuff = _block_data->allocate_io_buffer(BLOCK_SIZE, BLOCK_SIZE, NUMA_NODE_ANY);
  struct inode_t * inode = static_cast<struct inode_t*>(_block_data->virt_addr(iobuff));
  inode->magic = BLOB_INODE_MAGIC;
  inode->flags = 0;
  inode->nblocks = n_blocks;
  inode->size = size_in_bytes;
  inode->next_inode = 0;
  memset(inode->ranges,0,sizeof(inode->ranges));
  inode->ranges[0] = {dr->addr(), dr->addr() + n_blocks - 1};

  _block_data->write(iobuff, 0, hdr_lba, 1);
  _block_data->free_io_buffer(iobuff);

  PLOG("created blob@ %lx size=%ld blocks", hdr_lba, n_blocks);
  return reinterpret_cast<blob_t>(new Blob_handle(dr));
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
  Blob_handle * h = reinterpret_cast<Blob_handle*>(handle);
  {
    std::lock_guard<std::mutex> g(_md_lock);
    _data_allocator->free(h->dr->addr()); /* TODO optimize since we have Memory_region */
    flush_md();
  }
  delete h;
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
  Blob_handle * h = reinterpret_cast<Blob_handle*>(handle);
  if(size_in_bytes > h->dr->size())
    throw API_exception("truncation larger than existing size");

  /* now we have to trim the storage blocks */
  Inode inode(_block_data, h->dr->addr());

  assert(inode.size() >= size_in_bytes);

  if(inode.size() == size_in_bytes) return; /* no need to trim */

  size_t new_nblocks = (size_in_bytes / BLOCK_SIZE) + 1; //bytes_to_trim = inode.size() - size_in_bytes;
  size_t blocks_to_remove = inode.nblocks() - new_nblocks;

  while(blocks_to_remove > 0) {
    range_t * r = inode.last_range();
    if(r==nullptr) throw Logic_exception("truncating blocks");

    size_t delta = (r->last - r->first);
    if(blocks_to_remove >= delta) {
      /* remove complete range */
      r->last = r->first = 0;
      blocks_to_remove -= delta;
    }
    else {
      /* shrink range */
      r->last -= blocks_to_remove;
      blocks_to_remove = 0;
    }
  }

  assert(size_in_bytes <= new_nblocks * BLOCK_SIZE);
  /* update allocator */
  {
    std::lock_guard<std::mutex> g(_md_lock);
    _data_allocator->trim(h->dr->addr(), new_nblocks * BLOCK_SIZE);
  }

}

void
Blob_component::show_state(std::string filter)
{
  
  if(filter.compare("*")==0) {
    PINF("-- BLOB STORE STATE --");
    _data_allocator->apply([](addr_t addr,size_t size, bool free, bool head)
                           {
                             if(free) return;
                             
                             if(head) {
                               PINF("Head: %08lx-%08lx",addr,addr+size);
                             }
                           });
    PINF("-- end ---------------");
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

