#include "server_grouped_connection.h"

#include "eyecatcher.h"
#include <api/fabric_itf.h> /* IFabric_server_grouped_factory */
#include <exception>
#include <iostream> /* cerr */

Component::IFabric_server_grouped *server_grouped_connection::get_connection(Component::IFabric_server_grouped_factory &ep_)
{
  Component::IFabric_server_grouped *cnxn = nullptr;
  while ( ! ( cnxn = ep_.get_new_connection() ) ) {}
  return cnxn;
}

server_grouped_connection::server_grouped_connection(Component::IFabric_server_grouped_factory &ep_)
  : _ep(ep_)
  , _cnxn(get_connection(_ep))
  , _comm(_cnxn->allocate_group())
{
}
server_grouped_connection::~server_grouped_connection()
try
{
  delete _comm;
  _ep.close_connection(_cnxn);
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}

Component::IFabric_communicator *server_grouped_connection::allocate_group() const
{
  return cnxn().allocate_group();
}
