#include "fs_minix_component.h"

Component::IBlock_device * g_block_layer;


extern "C" int do_run_fuse();

namespace comanche {

Minix_fs_component::Minix_fs_component()
{
  
}

Minix_fs_component::~Minix_fs_component()
{
}


int Minix_fs_component::bind(IBase * component)
{
  using namespace Component;
  
  g_block_layer = _lower_layer = (IBlock_device*) component->query_interface(IBlock_device::iid());
  
  if(option_DEBUG && _lower_layer)
    PLOG("Minix_fs_component has bound to lower block device layer");

  if(_lower_layer) {
    _lower_layer->add_ref();
    return 0;
  }
  return 1;
}

status_t Minix_fs_component::start()
{
  if(_lower_layer == nullptr) {
    PERR("Minix_fs_component::start failed. no lower layer binding.");
    return E_FAIL;
  }

  PLOG("running fuse..");
  do_run_fuse();

  return S_OK;
}

status_t Minix_fs_component::stop()
{
  return E_NOT_IMPL;
}



}  // comanche

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  using namespace comanche;
  
  if(component_id == Minix_fs_component::component_id()) {
    printf("Creating 'Minix_fs' component.\n");
    return static_cast<void*>(new comanche::Minix_fs_component());
  }
  else return NULL;
}
