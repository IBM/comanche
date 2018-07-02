#include "server_connection.h"

#include "eyecatcher.h"
#include <api/fabric_itf.h> /* IFabric_server_factory, IFabric_server */
#include <exception>
#include <iostream> /* cerr */

Component::IFabric_server *server_connection::get_connection(Component::IFabric_server_factory &ep_)
{
  Component::IFabric_server *cnxn = nullptr;
  while ( ! ( cnxn = ep_.get_new_connections() ) ) {}
  return cnxn;
}

server_connection::server_connection(Component::IFabric_server_factory &ep_)
  : _ep(ep_)
  , _cnxn(get_connection(_ep))
{
}

server_connection::~server_connection()
try
{
  _ep.close_connection(_cnxn);
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
