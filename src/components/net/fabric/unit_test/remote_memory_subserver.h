#ifndef _TEST_REMOTE_MEMORY_SUBSERVER_H_
#define _TEST_REMOTE_MEMORY_SUBSERVER_H_

#include <common/types.h> /* status_t */
#include "registered_memory.h"
#include <api/fabric_itf.h> /* IFabric_commuicator */
#include <memory> /* shared_ptr */

class server_grouped_connection;

class remote_memory_subserver
{
  static void check_complete_static(void *rmc_, ::status_t stat_);
  static void check_complete_static_2(void *t_, void *rmc_, ::status_t stat);
public:
  void check_complete(::status_t stat_);
private:
  server_grouped_connection &_parent;
  std::shared_ptr<Component::IFabric_communicator> _cnxn;
  registered_memory _rm_out;
  registered_memory _rm_in;
  registered_memory &rm_out() { return _rm_out; }
  registered_memory &rm_in () { return _rm_in; }
public:
  remote_memory_subserver(server_grouped_connection &parent_);

  Component::IFabric_communicator &cnxn() { return *_cnxn; }
};

#endif
