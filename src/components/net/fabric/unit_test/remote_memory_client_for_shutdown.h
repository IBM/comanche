#ifndef _TEST_REMOTE_MEMORY_FOR_SHUTDOWN_H_
#define _TEST_REMOTE_MEMORY_FOR_SHUTDOWN_H_

#include "remote_memory_client.h"

#include <cstdint> /* uint64_t */
#include <cstring> /* string */

namespace Component
{
  class IFabric;
}

class remote_memory_client_for_shutdown
  : private remote_memory_client
{
public:
  remote_memory_client_for_shutdown(Component::IFabric &fabric, const std::string &fabric_spec, const std::string ip_address, std::uint16_t port);
};

#endif
