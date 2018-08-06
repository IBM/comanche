#ifndef _TEST_REMOTE_MEMORY_CLIENT_H_
#define _TEST_REMOTE_MEMORY_CLIENT_H_

#include "remote_memory_accessor.h"

#include <common/types.h> /* status_t */
#include <cstdint> /* uint16_t, uint64_t */
#include <string>
#include <memory> /* shared_ptr */

namespace Component
{
  class IFabric;
  class IFabric_client;
  class IFabric_communicator;
}

class registered_memory;

class remote_memory_client
  : public remote_memory_accessor
{
  static void check_complete_static(void *t, void *ctxt, ::status_t stat);
  void check_complete(::status_t stat);

  std::shared_ptr<Component::IFabric_client> _cnxn;
  std::shared_ptr<registered_memory> _rm_out;
  std::shared_ptr<registered_memory> _rm_in;
  std::uint64_t _vaddr;
  std::uint64_t _key;
  char _quit_flag;
  ::status_t _last_stat;

  registered_memory &rm_in() const { return *_rm_in; }
  registered_memory &rm_out() const { return *_rm_out; }
protected:
  void do_quit();
public:
  remote_memory_client(
    Component::IFabric &fabric
    , const std::string &fabric_spec
    , const std::string ip_address
    , std::uint16_t port
    , std::uint64_t remote_key_base
  );
  remote_memory_client(remote_memory_client &&) = default;
  remote_memory_client &operator=(remote_memory_client &&) = default;

  ~remote_memory_client();

  void send_disconnect(Component::IFabric_communicator &cnxn_, registered_memory &rm_, char quit_flag_);

  std::uint64_t vaddr() const { return _vaddr; }
  std::uint64_t key() const { return _key; }
  Component::IFabric_client &cnxn() { return *_cnxn; }

  void write(const std::string &msg_);

  void read_verify(const std::string &msg_);
};

#endif
