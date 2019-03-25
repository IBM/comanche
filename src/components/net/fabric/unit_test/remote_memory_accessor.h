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
#ifndef _TEST_REMOTE_MEMORY_ACCESSOR_H_
#define _TEST_REMOTE_MEMORY_ACCESSOR_H_

#include <cstddef> /* size_t */

namespace Component
{
  class IFabric_communicator;
}

class registered_memory;

class remote_memory_accessor
{
protected:
  void send_memory_info(Component::IFabric_communicator &cnxn, registered_memory &rm);
public:
  /* using rm as a buffer, send message */
  void send_msg(Component::IFabric_communicator &cnxn, registered_memory &rm, const void *msg, std::size_t len);
};

#endif
