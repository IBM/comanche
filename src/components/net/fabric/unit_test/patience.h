#ifndef _TEST_PATIENCE_H_
#define _TEST_PATIENCE_H_

#include <cstdint> /* uint16_t */
#include <string>

namespace Component
{
  class IFabric;
  class IFabric_client;
  class IFabric_client_grouped;
}

Component::IFabric_client *open_connection_patiently(Component::IFabric &fabric, const std::string &fabric_spec, const std::string ip_address, std::uint16_t port);

Component::IFabric_client_grouped *open_connection_grouped_patiently(Component::IFabric &fabric, const std::string &fabric_spec, const std::string ip_address, std::uint16_t port);

#endif
