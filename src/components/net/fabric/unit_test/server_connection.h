#ifndef _TEST_SERVER_CONNECTION_H_
#define _TEST_SERVER_CONNECTION_H_

#include "delete_copy.h"

namespace Component
{
  class IFabric_server;
  class IFabric_server_factory;
}

class server_connection
{
  Component::IFabric_server_factory &_ep;
  Component::IFabric_server *_cnxn;
  server_connection(server_connection &&) noexcept;
  DELETE_COPY(server_connection);
  static Component::IFabric_server *get_connection(Component::IFabric_server_factory &ep);
public:
  Component::IFabric_server &cnxn() const { return *_cnxn; }
  explicit server_connection(Component::IFabric_server_factory &ep);
  /* The presence of a destructor and a pointer member causes -Weffc++ to warn
   *
   * warning: ‘class d’ has pointer data members [-Weffc++]
   * warning:   but does not override ‘d(const d&)’ [-Weffc++]
   * warning:   or ‘operator=(const d&)’ [-Weffc++]
   *
   * g++ should not warn in this case, because the declarataion of a move constructor suppresses
   * default copy constructor and operator=.
   */
  ~server_connection();
};

#endif
