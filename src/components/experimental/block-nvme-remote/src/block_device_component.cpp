#include "block_device_component.h"

using namespace Component;

Block_device_component::Block_device_component(const char * json_config_file)
{
  assert(json_config_file);
  
  _dev = new Block_service(json_config_file);

  Json_configuration cfg(json_config_file);
  assert(cfg.volume_id());
  
  _lv = _dev->create_volume(cfg.volume_id());
  _dev->attach_local_device(_lv,0);  
  _session = _dev->create_session(_lv,1);
}

Block_device_component::~Block_device_component()
{
  delete _dev;
}


io_buffer_t
Block_device_component::
allocate_io_buffer(size_t size, unsigned alignment, int numa_node)
{
  if(size == 0) 
    throw API_exception("allocate_io_buffer: invalid operand");

  if(size % _lv->block_size() > 0)
    throw API_exception("size must be modulo block size of device");
    
  return _session->allocate_io_buffer(size, alignment, numa_node);
}

status_t
Block_device_component::
realloc_io_buffer(io_buffer_t io_mem, size_t size, unsigned alignment)
{
  if(size == 0) 
    throw API_exception("allocate_io_buffer: invalid operand");
  
  if(size % _lv->block_size() > 0)
    throw API_exception("size must be modulo block size of device");

  
}

status_t
Block_device_component::
free_io_buffer(io_buffer_t io_mem)
{
  return _session->free_io_buffer(io_mem);
}

io_buffer_t
Block_device_component::
register_memory_for_io(void * vaddr, size_t len)
{
  _session->register_memory_for_io(vaddr, len);
  // return vaddr;
  return 0;
}

void
Block_device_component::
unregister_memory_for_io(io_buffer_t buffer)
{
  // TO FIX
  //  _session->unregister_memory_for_io(vaddr, len);
}

void *
Block_device_component::
virt_addr(io_buffer_t buffer)
{
  return _session->virt_addr(buffer);
}

addr_t
Block_device_component::
phys_addr(io_buffer_t buffer)
{
  return _session->phys_addr(buffer);
}

workid_t
Block_device_component::
async_read(io_buffer_t buffer,
           uint64_t offset,
           uint64_t lba,
           uint64_t lba_count)
{
  return _session->async_submit(COMANCHE_OP_READ, buffer, offset, lba, lba_count);
}

workid_t
Block_device_component::
async_write(io_buffer_t buffer,
            uint64_t offset,
            uint64_t lba,
            uint64_t lba_count)
{
  return _session->async_submit(COMANCHE_OP_WRITE, buffer, offset, lba, lba_count);
}

bool
Block_device_component::
check_completion(uint64_t gwid)
{
  return _session->check_completion(gwid);
}

void
Block_device_component::
get_volume_info(VOLUME_INFO& devinfo)
{
  _session->get_volume_info(devinfo);

  /* get other volume information from logical volume */
  strncpy(devinfo.volume_name,
          _lv->name(),
          VOLUME_INFO_MAX_NAME);
}


namespace Component {

Component::IBlock_device* create_block_device(const char * json_configuration)
{
  return static_cast<Component::IBlock_device *>
    (new Block_device_component(json_configuration));
}

void release_block_device(Component::IBlock_device* block_dev)
{
  delete (static_cast<Block_device_component *>(block_dev));
}

} /*< Component */



// C only wrapper

/** 
 * Create block device instance
 * 
 * @param json_configuration 
 * 
 * @return handle to block device instance
 */
extern "C"
handle_t block_device_create(const char * json_configuration)
{
  handle_t h;
  try {
    h = static_cast<handle_t>(create_block_device(json_configuration));
  }
  catch(...) {
    return 0;
  }
  return h;
}

/** 
 * Release block device
 * 
 * @param handle block device instance
 */
extern "C"
status_t block_device_release(handle_t handle)
{
  try {
    IBlock_device* bdev = static_cast<IBlock_device*>(handle);
    release_block_device(bdev);
  }
  catch(...) {
    return E_FAIL;
  }
  return S_OK;
}

/** 
 * Allocate a contiguous memory region that can be used for IO
 * 
 * @param block device handle
 * @param size Size of memory in bytes
 * @param alignment Alignment
 * 
 * @return Handle to IO memory region
 */
extern "C"
io_buffer_t block_device_allocate_io_buffer(handle_t handle, size_t size, size_t alignment)
{
  try {
    assert(handle);
    IBlock_device* bdev = static_cast<IBlock_device*>(handle);
    return bdev->allocate_io_buffer(size, alignment, -1);
  }
  catch(...) {
    return 0;
  }
}

/** 
 * Free a previously allocated buffer
 * 
 * @param block device handle
 * @param io_mem Handle to IO memory allocated by allocate_io_buffer
 * 
 * @return S_OK on success
 */
extern "C"
status_t block_device_free_io_buffer(handle_t handle, io_buffer_t io_mem)
{
  assert(handle);
  IBlock_device* bdev = static_cast<IBlock_device*>(handle);
  return bdev->free_io_buffer(io_mem);
}

/** 
 * Register memory for DMA with the SPDK subsystem.
 * 
 * @param block device handle
 * @param vaddr 
 * @param len 
 */
extern "C"
io_buffer_t block_device_register_memory_for_io(handle_t handle, void * vaddr, size_t len)
{
  assert(handle);
  IBlock_device* bdev = static_cast<IBlock_device*>(handle);
  return bdev->register_memory_for_io(vaddr, len);
}

/** 
 * Unregister memory for DMA with the SPDK subsystem.
 * 
 * @param block device handle
 * @param vaddr 
 * @param len 
 */
extern "C"
void unregister_memory_for_io(handle_t handle, io_buffer_t buffer)
{
  assert(handle);
  IBlock_device* bdev = static_cast<IBlock_device*>(handle);
  bdev->unregister_memory_for_io(buffer);
}

/** 
 * Get pointer (virtual address) to start of IO buffer
 * 
 * @param block device handle
 * @param buffer IO buffer handle
 * 
 * @return pointer or NULL on error
 */
extern "C"
void * block_device_virt_addr(handle_t handle, io_buffer_t buffer)
{
  try {
    assert(handle);
    IBlock_device* bdev = static_cast<IBlock_device*>(handle);
    return bdev->virt_addr(buffer);
  }
  catch(...) {
    return nullptr;
  }
}

/** 
 * Get physical address of buffer
 * 
 * @param block device handle
 * @param buffer IO buffer handle
 * 
 * @return physical address or 0 on error
 */
extern "C"
addr_t block_device_phys_addr(handle_t handle, io_buffer_t buffer)
{
  try {
    assert(handle);
    IBlock_device* bdev = static_cast<IBlock_device*>(handle);
    return bdev->phys_addr(buffer);
  }
  catch(...) {
    return 0;
  }
}

/** 
 * Submit asynchronous read operation
 * 
 * @param block device handle
 * @param buffer IO buffer
 * @param lba block address
 * @param lba_count number of blocks
 * 
 * @return work identifier
 */
extern "C"
workid_t block_device_async_read(handle_t handle, io_buffer_t buffer, uint64_t buffer_offset, uint64_t lba, uint64_t lba_count)
{
  assert(handle);
  IBlock_device* bdev = static_cast<IBlock_device*>(handle);
  return bdev->async_read(buffer, buffer_offset, lba, lba_count);
}

/** 
 * Submit asynchronous write operation
 * 
 * @param block device handle
 * @param buffer IO buffer
 * @param lba logical block address
 * @param lba_count number of blocks
 * 
 * @return work identifier
 */
extern "C"
workid_t block_device_async_write(handle_t handle, io_buffer_t buffer, uint64_t buffer_offset, uint64_t lba, uint64_t lba_count)
{
  assert(handle);
  IBlock_device* bdev = static_cast<IBlock_device*>(handle);
  return bdev->async_write(buffer, buffer_offset, lba, lba_count);
}

/** 
 * Check for completion of a work request. This API is thread-safe.
 * 
 * @param block device handle
 * @param gwid Work request identifier
 * 
 * @return True if completed.
 */
extern "C"
int block_device_check_completion(handle_t handle, workid_t gwid)
{
  assert(handle);
  IBlock_device* bdev = static_cast<IBlock_device*>(handle);
  return bdev->check_completion(gwid);
}



/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Block_device_component_factory::component_id()) {
    printf("Creating 'Block_device_factory' component.\n");
    return static_cast<void*>(new Block_device_component_factory());
  }
  else return NULL;
}
