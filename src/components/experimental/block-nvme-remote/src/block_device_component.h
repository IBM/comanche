#ifndef __COMANCHE_BLOCK_DEVICE_H__
#define __COMANCHE_BLOCK_DEVICE_H__

#include <mutex>
#include <block_service.h>
#include <api/block_itf.h>

using namespace Component;

/** 
 * Block_device: provides unified API for NVMe device access.
 * Incorporates Block_service and Block_service_session.  Current
 * version is simple, and supports a single session. 
 * 
 */
class Block_device_component: public Component::IBlock_device
{
public:

  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x24518582,0xd731,0x4406,0x9eb1,0x58,0x70,0x26,0x40,0x8e,0x23);

  void * query_interface(Component::uuid_t& itf_uuid) {
    if(itf_uuid == IBlock_device::iid()) {
      return (void *) static_cast<IBlock_device *>(this);
    }
    else 
      return NULL; // we don't support this interface
  };

  void unload() {
    delete this;
  }
  
  DUMMY_IBASE_CONTROL;
  
public:
  
  /** 
   * Constructor
   * 
   */
  explicit Block_device_component(const char * json_config_file);

  /**
   * Default destructor
   */
  virtual ~Block_device_component() noexcept;


  /** 
   * Allocate a memory region that can be used for IO
   * 
   * @param size Size of memory in bytes
   * @param alignment Alignment
   * @param numa_node NUMA node specifier
   * 
   * @return Handle to IO memory region
   */
  virtual io_buffer_t allocate_io_buffer(size_t size, unsigned alignment, int numa_node);

  /** 
   * Re-allocate area of memory
   * 
   * @param io_mem Memory handle (from allocate_io_buffer)
   * @param size New size of memory in bytes
   * @param alignment Alignment in bytes
   * 
   * @return S_OK or E_NO_MEM (unable) or E_NOT_IMPL
   */
  virtual status_t realloc_io_buffer(io_buffer_t io_mem, size_t size, unsigned alignment);
  
  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  virtual status_t free_io_buffer(io_buffer_t io_mem);

  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual io_buffer_t register_memory_for_io(void * vaddr, size_t len);

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void unregister_memory_for_io(io_buffer_t buffer);

  /** 
   * Get pointer to start of IO buffer
   * 
   * @param buffer IO buffer
   * 
   * @return pointer
   */
  virtual void * virt_addr(io_buffer_t buffer);

  /** 
   * Get physical address of buffer
   * 
   * @param buffer IO buffer handle
   * 
   * @return physical address
   */
  virtual addr_t phys_addr(io_buffer_t buffer);

  /** 
   * Submit asynchronous read operation
   * 
   * @param buffer IO buffer
   * @param lba logical block address
   * @param lba_count logical block count
   * 
   * @return work identifier
   */
  virtual workid_t async_read(io_buffer_t buffer, uint64_t offset, uint64_t lba, uint64_t lba_count);

  /** 
   * Submit asynchronous write operation
   * 
   * @param buffer IO buffer
   * @param lba logical block address
   * @param lba_count logical block count
   * 
   * @return work identifier
   */
  virtual workid_t async_write(io_buffer_t buffer, uint64_t offset, uint64_t lba, uint64_t lba_count);

  
  /** 
   * Check for completion of a work request. This API is thread-safe.
   * 
   * @param gwid Work request identifier
   * 
   * @return True if completed.
   */
  virtual bool check_completion(uint64_t gwid);


  /** 
   * Get device information
   * 
   * @param devinfo pointer to VOLUME_INFO struct
   * 
   * @return S_OK on success
   */
  virtual void get_volume_info(Component::VOLUME_INFO& devinfo);

 
private:
  
  Logical_volume *             _lv;
  Block_service *              _dev;
  Block_service_session*       _session;
};

/** 
 * Factory for Block_device component
 * 
 */
class Block_device_component_factory : public Component::IBlock_device_factory
{
public:
  DECLARE_COMPONENT_UUID(0xFAC2215b,0xff57,0x4efd,0x9e4e,0xca,0xe8,0x9b,0x70,0x73,0x2a);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == IBlock_device_factory::iid()) {
      return (void *) static_cast<IBlock_device_factory *>(this);
    }
    else 
      return NULL; // we don't support this interface
  };

  void unload() override {
    delete this;
  }
  
  IBlock_device * create(std::string config_filename) override {
    IBlock_device * inst = static_cast<Component::IBlock_device *>
      (new Block_device_component(config_filename.c_str()));
    inst->add_ref();
    return inst;
  }

  DUMMY_IBASE_CONTROL;
};



#endif
