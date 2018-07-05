#ifndef _TEST_REMOTE_MEMORY_CLIENT_GROUPED_H_
#define _TEST_REMOTE_MEMORY_CLIENT_GROUPED_H_

#include "remote_memory_accessor.h"

#include <common/types.h> /* status_t */
#include <cstdint> /* uint16_t */
#include <cstring> /* string */
#include <memory> /* shared_ptr */

class registered_memory;

namespace Component
{
  class IFabric;
  class IFabric_client_grouped;
  class IFabric_communicaator;
}

class remote_memory_client_grouped
  : public remote_memory_accessor
{
  std::shared_ptr<Component::IFabric_client_grouped> _cnxn;
  std::shared_ptr<registered_memory> _rm_out;
  std::shared_ptr<registered_memory> _rm_in;
  std::uint64_t _vaddr;
  std::uint64_t _key;
  char _quit_flag;

  registered_memory &rm_in() const { return *_rm_in; }
  registered_memory &rm_out() const { return *_rm_out; }

  static void check_complete_static(void *rmc, ::status_t stat);
  static void check_complete_static_2(void *t, void *rmc, ::status_t stat);
  static void check_complete(::status_t stat);

public:
  remote_memory_client_grouped(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_);

  remote_memory_client_grouped(remote_memory_client_grouped &&) = default;
  remote_memory_client_grouped &operator=(remote_memory_client_grouped &&) = default;

  ~remote_memory_client_grouped();

  void send_disconnect(Component::IFabric_communicator &cnxn, registered_memory &rm, char quit_flag);

  std::uint64_t vaddr() const { return _vaddr; }
  std::uint64_t key() const { return _key; }
  Component::IFabric_client_grouped &cnxn() { return *_cnxn; }

  Component::IFabric_communicator *allocate_group() const;
};

#endif
