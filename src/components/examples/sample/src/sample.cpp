#include <iostream>
#include "sample.h"

Sample::Sample(std::string name) : _name(name)
{
}

Sample::~Sample()
{
}

void Sample::say_hello()
{
  std::cout << "Hello " << _name << std::endl;
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Sample_factory::component_id()) {
    return static_cast<void*>(new Sample_factory());
  }
  else return NULL;
}

#undef RESET_STATE
