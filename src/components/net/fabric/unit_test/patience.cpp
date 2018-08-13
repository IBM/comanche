#include "patience.h"

#include <gtest/gtest.h>
#include <api/fabric_itf.h> /* IFabric, IFabric_client, IFabric_client_grouped */
#include <system_error>

Component::IFabric_client *open_connection_patiently(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
{
  Component::IFabric_client *cnxn = nullptr;
  int try_count = 0;
  while ( ! cnxn )
  {
    try
    {
      cnxn = fabric_.open_client(fabric_spec_, ip_address_, port_);
    }
    catch ( std::system_error &e )
    {
      if ( e.code().value() != ECONNREFUSED )
      {
        throw;
      }
    }
    ++try_count;
  }
  EXPECT_LT(0U, cnxn->max_message_size());
  return cnxn;
}

Component::IFabric_client_grouped *open_connection_grouped_patiently(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
{
  Component::IFabric_client_grouped *cnxn = nullptr;
  int try_count = 0;
  while ( ! cnxn )
  {
    try
    {
      cnxn = fabric_.open_client_grouped(fabric_spec_, ip_address_, port_);
    }
    catch ( std::system_error &e )
    {
      if ( e.code().value() != ECONNREFUSED )
      {
        throw;
      }
    }
    ++try_count;
  }
  EXPECT_LT(0U, cnxn->max_message_size());
  return cnxn;
}
