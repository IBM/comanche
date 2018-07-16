#ifndef _TEST_SERVER_CONNECTION_AND_MEMORY_H_
#define _TEST_SERVER_CONNECTION_AND_MEMORY_H_

#include "server_connection.h"
#include "registered_memory.h"
#include "remote_memory_accessor.h"
#include <boost/core/noncopyable.hpp>
#include <cstdint> /* uint64_t */

namespace Component
{
  class IFabric_server_factory;
}

class server_connection_and_memory
  : public server_connection
  , public registered_memory
  , public remote_memory_accessor
  , private boost::noncopyable
{
public:
  server_connection_and_memory(Component::IFabric_server_factory &ep, std::uint64_t remote_key);
  ~server_connection_and_memory();
};

#endif
