#include <common/exceptions.h>
#include <core/zerocopy_passthrough.h>
#include <string.h>
#include "part_region_component.h"

/** 
 * Represents a window onto the block device defined by the partition
 * 
 */
class Region_session : public Core::Zerocopy_passthrough_impl<Component::IBlock_device>

{
public:
  Region_session(Component::IBlock_device *lower_layer,
                 std::string volume_name,
                 addr_t first_lba,
                 addr_t last_lba) :
    _first_lba(first_lba),
    _last_lba(last_lba),
    _volume_name(volume_name)
  {    
    PINF("Region session (%s) %lu-%lu (lower=%p)", volume_name.c_str(), first_lba, last_lba, lower_layer);
    
    if(last_lba < first_lba)
      throw Constructor_exception("invalid params (last_lba < first_lba)");
    
    Zerocopy_passthrough_impl<Component::IBlock_device>::_lower_layer = lower_layer;
    lower_layer->add_ref();
    
    assert(last_lba > 0);

    _capacity = _last_lba - _first_lba + 1;

    this->add_ref(); /* add initial reference */
  }

  virtual ~Region_session() {
  }
  
  void * query_interface(Component::uuid_t& itf) override {
    return nullptr;
  }

  void unload() override {
    PLOG("unloading part-region component (%p)", this);
    delete this;
  }

  Component::workid_t async_read(Component::io_buffer_t buffer,
                                 uint64_t buffer_offset,
                                 uint64_t lba,
                                 uint64_t lba_count,
				 int queue_id = 0,
                                 io_callback_t cb = nullptr,
                                 void * cb_arg0 = nullptr,
                                 void * cb_arg1 = nullptr) override {
    if((lba + lba_count - 1) > _capacity)
      throw API_exception("async_read: out of bounds");
    return _lower_layer->async_read(buffer,
                                    buffer_offset,
                                    lba + _first_lba, /* add offset */
                                    lba_count,
				    queue_id,
                                    cb,
                                    cb_arg0,
                                    cb_arg1);
  }
  
  Component::workid_t async_write(Component::io_buffer_t buffer,
                                  uint64_t buffer_offset,
                                  uint64_t lba,
                                  uint64_t lba_count,
				  int queue_id = 0,
                                  io_callback_t cb = nullptr,
                                  void * cb_arg0 = nullptr,
                                  void * cb_arg1 = nullptr) override {
    if((lba + lba_count - 1) > _capacity)
      throw API_exception("async_write: out of bounds");
    
    return _lower_layer->async_write(buffer,
                                     buffer_offset,
                                     lba + _first_lba, /* add offset */
                                     lba_count,
				     queue_id,
                                     cb,
                                     cb_arg0,
                                     cb_arg1);
  }

  void read(Component::io_buffer_t buffer,
            uint64_t buffer_offset,
            uint64_t lba,
            uint64_t lba_count,
	    int queue_id = 0) override {
    _lower_layer->read(buffer, buffer_offset, lba + _first_lba, lba_count, queue_id);
  }

  void write(Component::io_buffer_t buffer,
             uint64_t buffer_offset,
             uint64_t lba,
             uint64_t lba_count,
	     int queue_id = 0) override {
    _lower_layer->write(buffer, buffer_offset, lba + _first_lba, lba_count, queue_id);
  }

  
  bool check_completion(Component::workid_t gwid, int queue_id = 0) override {
    return _lower_layer->check_completion(gwid, queue_id);
  }

  void get_volume_info(Component::VOLUME_INFO& devinfo) override {
    _lower_layer->get_volume_info(devinfo);
    std::stringstream ss;
    ss << devinfo.volume_name << ":Region(" << _first_lba << "-" << _last_lba << ")";
    
    strcpy(devinfo.volume_name, ss.str().c_str());
    devinfo.block_count = _capacity;
  }

private:
  std::string _volume_name;
  addr_t      _first_lba, _last_lba;
  size_t      _capacity;
};

/* factory */

Component::IRegion_manager *
Region_manager_factory::
open(Component::IBlock_device * block_device, int flags)
{
  if(block_device == nullptr)
    throw API_exception("bad block interface");

  bool force_init = flags & IRegion_manager_factory::FLAGS_FORMAT;
  PLOG("Region_manager_factory: flags=%x", force_init);
  
  auto obj = static_cast<Component::IRegion_manager *>
    (new Region_manager(block_device, force_init));

  obj->add_ref();
  return obj;
}

/* main component */

Region_manager::Region_manager(Component::IBlock_device * block_device,
                               bool force_init) :
  _lower_block_layer(block_device),
  _region_table(block_device, force_init)
{
  if(block_device==nullptr)
    throw Constructor_exception("invalid parameter");
  
  block_device->get_volume_info(_vi);
}

Region_manager::~Region_manager()
{
}


/** 
 * Re-use or allocate a region of space
 * 
 * @param size Size of region in blocks
 * @param alignment Alignment in bytes
 * @param owner Owner
 * @param id Identifier
 * @param reused [out] true if re-used.
 * 
 * @return Block device interface onto region
 */
Component::IBlock_device *
Region_manager::
reuse_or_allocate_region(size_t size_in_blocks,
                         std::string owner,
                         std::string id,
                         addr_t& vaddr,
                         bool& reused)
{
  __region_desc_t * rd = _region_table.find(owner, id);
  if(rd) {
    reused = true;

    if(option_DEBUG) {
      _region_table.dump();
      PLOG("Using existing region '%s'",id.c_str());
    }
  }
  else {
    if(option_DEBUG)
      PLOG("Creating new region '%s' of %lu blocks",id.c_str(), size_in_blocks);
    
    rd = _region_table.allocate(size_in_blocks,
                                1, /*< alignment */
                                owner,
                                id);
    if(rd==nullptr)
      throw General_exception("region table allocation failed");
    
    reused = false;
  }
  vaddr = rd->vaddr;

  return new Region_session(_lower_block_layer,
                            id.c_str(),
                            rd->saddr,
                            rd->saddr + rd->size - 1);
}


/** 
 * Retrieve region information
 * 
 * @param index Index counting from 0
 * @param ri REGION_INFO [out]
 * 
 * @return true if valid and REGION_INFO has been filled
 */
bool
Region_manager::
get_region_info(unsigned index, REGION_INFO& ri)
{
  __region_desc_t * rd = _region_table.get_entry(index);
  if(rd == nullptr) return false;
  ri.block_size = _vi.block_size;
  ri.size_in_blocks = rd->size;
  return true;
}

bool
Region_manager::
get_region_info(std::string owner, std::string id, REGION_INFO& ri)
{
  __region_desc_t * rd = _region_table.find(owner, id);
  if(rd == nullptr || rd->occupied == false) return false;
  ri.block_size = _vi.block_size;
  ri.size_in_blocks = rd->size;
  return true;
}

/** 
 * Delete a region
 * 
 * @param index Region index
 */
bool Region_manager::delete_region(std::string owner, std::string id)
{
  return _region_table.remove_entry(owner, id);
}

size_t Region_manager::block_size()
{
  return _vi.block_size;
}

size_t Region_manager::num_regions()
{
  return _region_table.num_allocated_entries();
}

void Region_manager::get_underlying_volume_info(Component::VOLUME_INFO& vi)
{
  _lower_block_layer->get_volume_info(vi);
}



/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Region_manager_factory::component_id()) {
    PLOG("Creating 'Region_manager' factory.");
    return static_cast<void*>(new Region_manager_factory());
  }
  else
    return NULL;
}

