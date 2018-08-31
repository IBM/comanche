#include "remote_memory_client_for_shutdown.h"

remote_memory_client_for_shutdown::remote_memory_client_for_shutdown(
  Component::IFabric &fabric_
  , const std::string &fabric_spec_
  , const std::string ip_address_
  , std::uint16_t port_
  , std::uint64_t memory_size_
  , std::uint64_t remote_key_base_
)
  : remote_memory_client(fabric_, fabric_spec_, ip_address_, port_, memory_size_, remote_key_base_)
{
  do_quit();
}
