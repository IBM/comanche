#ifndef _TEST_SERVERC_GROUPED_CONNECTION_H_
#define _TEST_SERVERC_GROUPED_CONNECTION_H_

#include "delete_copy.h"

namespace Component
{
  class IFabric_server_grouped_factory;
  class IFabric_server_grouped;
  class IFabric_communicator;
}

class server_grouped_connection
{
  Component::IFabric_server_grouped_factory &_ep;
  /* ERROR: these two ought to be shared_ptr, with appropriate destructors */
  Component::IFabric_server_grouped *_cnxn;
  Component::IFabric_communicator *_comm;

  server_grouped_connection(server_grouped_connection &&) noexcept;
  DELETE_COPY(server_grouped_connection);
  static Component::IFabric_server_grouped *get_connection(Component::IFabric_server_grouped_factory &ep);

public:
  Component::IFabric_server_grouped &cnxn() const { return *_cnxn; }
  server_grouped_connection(Component::IFabric_server_grouped_factory &ep);
  ~server_grouped_connection();
  Component::IFabric_communicator &comm() const { return *_comm; }
  Component::IFabric_communicator *allocate_group() const;
};

#endif
