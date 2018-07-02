#ifndef _TEST_REMOTE_MEMORY_SUBCLIENT_H_
#define _TEST_REMOTE_MEMORY_SUBCLIENT_H_

#include <common/types.h> /* status_t */
#include "registered_memory.h"
#include <string>
#include <memory> /* shared_ptr */

namespace Component
{
  class IFabric_communicator;
}

class remote_memory_client_grouped;

class remote_memory_subclient
{
  static void check_complete_static(void *rmc, ::status_t stat);
  static void check_complete_static_2(void *t, void *rmc, ::status_t stat);
public:
  void check_complete(::status_t stat);
private:
  remote_memory_client_grouped &_parent;
  std::shared_ptr<Component::IFabric_communicator> _cnxn;
  registered_memory _rm_out;
  registered_memory _rm_in;
  registered_memory &rm_out() { return _rm_out; }
  registered_memory &rm_in () { return _rm_in; }
public:
  remote_memory_subclient(remote_memory_client_grouped &parent);

  Component::IFabric_communicator &cnxn() { return *_cnxn; }

  void write(const std::string &msg);
  void read_verify(const std::string &msg);
};

#endif
