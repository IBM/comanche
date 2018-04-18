#include <iostream>
#include "zyre_component.h"

Zyre_component::Zyre_component(const std::string& node_name)
{
}

Zyre_component::~Zyre_component()
{
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Zyre_component_factory::component_id()) {
    return static_cast<void*>(new Zyre_component_factory());
  }
  else return NULL;
}

