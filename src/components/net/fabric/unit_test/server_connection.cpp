#include "server_connection.h"

#include "eyecatcher.h"
#include <api/fabric_itf.h> /* IFabric_server_factory, IFabric_server */
#include <exception>
#include <iostream> /* cerr */

Component::IFabric_server *server_connection::get_connection(Component::IFabric_server_factory &ep_)
{
  Component::IFabric_server *cnxn = nullptr;
  while ( ! ( cnxn = ep_.get_new_connection() ) ) {}
  return cnxn;
}

server_connection::server_connection(Component::IFabric_server_factory &ep_)
  : _ep(&ep_)
  , _cnxn(get_connection(*_ep))
{
}

server_connection::server_connection(server_connection &&sc_) noexcept
  : _ep(sc_._ep)
  , _cnxn(std::move(sc_._cnxn))
{
  sc_._cnxn = nullptr;
}

server_connection &server_connection::operator=(server_connection &&sc_) noexcept
{
  _ep = sc_._ep;
  _cnxn = std::move(sc_._cnxn);
  sc_._cnxn = nullptr;
  return *this;
}

server_connection::~server_connection()
try
{
  if ( _cnxn )
  {
    _ep->close_connection(_cnxn);
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
