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
