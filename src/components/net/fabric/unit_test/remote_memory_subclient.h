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
  static void check_complete_static(void *t, void *ctxt, ::status_t stat);
  void check_complete(::status_t stat);

public:
  remote_memory_subclient(remote_memory_client_grouped &parent, std::uint64_t remote_key_index_);

  Component::IFabric_communicator &cnxn() { return *_cnxn; }

  void write(const std::string &msg);
  void read_verify(const std::string &msg);
};

#endif
