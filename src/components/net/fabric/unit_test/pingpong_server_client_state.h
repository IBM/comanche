#ifndef _TEST_PINGPONG_SERVER_CLIENT_STATE_H_
#define _TEST_PINGPONG_SERVER_N_H_

#include "registered_memory.h"
#include "server_connection.h"
#include <sys/uio.h> /* iovec */
#include <cstdint> /* uint64_t */
#include <vector>

namespace Component
{
  class IFabric_connection;
  class IFabric_server_factory;
}

struct client_state;

struct cb_ctxt
{
  client_state *state;
  using cb_t = void (*)(client_state *ctxt, ::status_t stat, std::uint64_t, std::size_t len, void *);
  static void cb(void *ctxt, ::status_t stat, std::uint64_t flags, std::size_t len, void *err);
  cb_t cb_call;
  explicit cb_ctxt(client_state *, cb_t);
};

struct buffer_state
{
  registered_memory _rm;
  std::vector<::iovec> v;
  std::vector<void *> d;
  explicit buffer_state(Component::IFabric_connection &cnxn, std::size_t size, std::uint64_t remote_key, std::size_t msg_size);
  buffer_state(buffer_state &&) = default;
  buffer_state &operator=(buffer_state &&) = default;
};

struct client_state
{
  cb_ctxt send_ctxt;
  cb_ctxt recv0_ctxt;
  cb_ctxt recv1_ctxt;
  server_connection sc;
  std::size_t msg_size;
  std::vector<buffer_state> br;
  buffer_state bt;
  unsigned iterations_left;
  explicit client_state(Component::IFabric_server_factory &factory_, std::size_t buffer_size_, std::uint64_t remote_key_, unsigned iteration_count_, std::size_t msg_size_);
  client_state(const client_state &) = delete;
  client_state(client_state &&cs_);
};

#endif
