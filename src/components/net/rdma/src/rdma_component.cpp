#include <iostream>
#include "rdma_component.h"

Rdma_component::Rdma_component(const std::string& device_name)
{
}

Rdma_component::~Rdma_component()
{
}

status_t Rdma_component::connect(const std::string& peer_name, int port)
{
  return E_FAIL;
}

status_t Rdma_component::disconnect()
{
  return E_FAIL;
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Rdma_component_factory::component_id()) {
    return static_cast<void*>(new Rdma_component_factory());
  }
  else return NULL;
}

