/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
