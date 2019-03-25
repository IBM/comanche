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
#ifndef _TEST_REMOTE_MEMORY_SUBCLIENT_H_
#define _TEST_REMOTE_MEMORY_SUBCLIENT_H_

#include <common/types.h> /* status_t */
#include "registered_memory.h"
#include <cstdint> /* uint64_t */
#include <memory> /* shared_ptr */
#include <string>

namespace Component
{
  class IFabric_communicator;
}

class remote_memory_client_grouped;

class remote_memory_subclient
{
  remote_memory_client_grouped &_parent;
  std::shared_ptr<Component::IFabric_communicator> _cnxn;
  registered_memory _rm_out;
  registered_memory _rm_in;

  registered_memory &rm_out() { return _rm_out; }
  registered_memory &rm_in () { return _rm_in; }
  static void check_complete_static(void *t, void *ctxt, ::status_t stat, std::size_t len);
  void check_complete(::status_t stat, std::size_t len);

public:
  remote_memory_subclient(
    remote_memory_client_grouped &parent
    , std::size_t memory_size
    , std::uint64_t remote_key_index
  );

  Component::IFabric_communicator &cnxn() { return *_cnxn; }

  void write(const std::string &msg);
  void read_verify(const std::string &msg);
};

#endif
