#include "remote_memory_client_for_shutdown.h"

remote_memory_client_for_shutdown::remote_memory_client_for_shutdown(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
  : remote_memory_client(fabric_, fabric_spec_, ip_address_, port_)
{
  do_quit();
}
