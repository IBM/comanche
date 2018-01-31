#include <common/exceptions.h>
#include <core/zerocopy_passthrough.h>
#include <string.h>
#include "part_gpt_component.h"
#include "gpt_types.h"
#include "gpt.h"


struct open_partition
{
  struct efi_entry efi_entry;
};


/** 
 * Represents a window onto the block device defined by the partition
 * 
 */
class Partition_session : public Core::Zerocopy_passthrough_impl<Component::IBlock_device>

{
public:
  Partition_session(Component::IBlock_device *lower_layer,
                    struct efi_entry * efi_entry)
  {
    Zerocopy_passthrough_impl<Component::IBlock_device>::_lower_layer = lower_layer;
    lower_layer->add_ref();
    
    PLOG("new Partition_session (%p)", this);
    memcpy(&_efi, efi_entry, sizeof(struct efi_entry));

    assert(_efi.last_lba > 0);
    assert(_efi.last_lba > _efi.first_lba);

    _capacity = _efi.last_lba - _efi.first_lba + 1;
  }

  virtual ~Partition_session() {
    PLOG("deleting Partition_session");
    _lower_layer->release_ref();
  }

  void release_ref() {
  }
  
  void * query_interface(Component::uuid_t& itf) {
    return nullptr;
  }

  Component::workid_t async_read(Component::io_buffer_t buffer,
                                 uint64_t buffer_offset,
                                 uint64_t lba,
                                 uint64_t lba_count,
				 int queue_id = 0,
                                 io_callback_t cb = nullptr,
                                 void * cb_arg0 = nullptr,
                                 void * cb_arg1 = nullptr)  {
    if((lba + lba_count + buffer_offset - 1) > _capacity)
      throw API_exception("async_read: out of bounds");

    return _lower_layer->async_read(buffer,
                                    buffer_offset,
                                    lba + _efi.first_lba, /* add offset */
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
                                  void *cb_arg0 = nullptr,
                                  void *cb_arg1 = nullptr) {
    if((lba + lba_count + buffer_offset - 1) > _capacity)
      throw API_exception("async_write: out of bounds");
    
    return _lower_layer->async_write(buffer,
                                     buffer_offset,
                                     lba + _efi.first_lba, /* add offset */
                                     lba_count,
				     queue_id,
                                     cb,
                                     cb_arg0,
                                     cb_arg1);
  }

  bool check_completion(Component::workid_t gwid, int queue_id = 0) {
    return _lower_layer->check_completion(gwid, queue_id);
  }

  void get_volume_info(Component::VOLUME_INFO& devinfo) {
    _lower_layer->get_volume_info(devinfo);
    devinfo.block_count = _capacity;

    char tmpname[EFI_NAMELEN + 1] = {0};    
    for(unsigned i=0;i<EFI_NAMELEN;i++)
      tmpname[i] = _efi.name[i] & 0xFF;
    
    strncpy(devinfo.volume_name, tmpname, Component::VOLUME_INFO_MAX_NAME);
  }

private:
  struct efi_entry _efi;
  size_t _capacity;
};

/* factory */

Component::IPartitioned_device *
GPT_component_factory::
create(Component::IBlock_device * block_device)
{
  if(block_device == nullptr)
    throw API_exception("bad block interface");

  auto obj = static_cast<Component::IPartitioned_device *>
    (new GPT_component(block_device));

  obj->add_ref();
  return obj;
}

/* main component */

GPT_component::GPT_component(Component::IBlock_device * block_device) :
  _table(block_device),
  _lower_block_layer(block_device)
{
  if(block_device==nullptr)
    throw Constructor_exception("invalid parameter");

  block_device->add_ref();
}

GPT_component::~GPT_component()
{
}



Component::IBlock_device *
GPT_component::
open_partition(unsigned partition_id)
{
  if(partition_id > _table.hdr()->entries_count)
    throw API_exception("partition id exceeded max count");

  
  struct efi_entry * e = _table.get_entry(partition_id);
  if(e->type_uuid[0] == 0)
    throw API_exception("invalid partition");

  /* TODO: check existing sessions? */  

  auto session = new Partition_session(_lower_block_layer, e);
  _sessions.push_front(session);
  
  return static_cast<Component::IBlock_device*>(session);
}

void
GPT_component::
release_partition(Component::IBlock_device * block_device)
{
  for(std::list<Partition_session*>::iterator i=_sessions.begin();
      i!=_sessions.end(); i++) {
    if(block_device == static_cast<Component::IBlock_device*>(*i)) {
      _sessions.erase(i);
      return;
    }
  }
  throw API_exception("invalid session on release_partition");
}

bool
GPT_component::
get_partition_info(unsigned partition_id,
                   size_t& size,
                   std::string& part_type,
                   std::string& description)
{
  return false;
}




/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  
  if(component_id == GPT_component_factory::component_id()) {
    PLOG("Creating 'GPT_component' factory.");
    return static_cast<void*>(new GPT_component_factory());
  }
  else
    return NULL;
}

