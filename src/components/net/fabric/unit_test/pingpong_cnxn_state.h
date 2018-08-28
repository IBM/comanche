#ifndef _TEST_PINGPONG_CNXN_STATE_H_
#define _TEST_PINGPONG_CNXN_STATE_H_

#include "registered_memory.h"
#include <sys/uio.h> /* iovec */
#include <cstdint> /* uint64_t */
#include <vector>

namespace Component
{
  class IFabric_connection;
  class IFabric_active_endpoint_comm;
}

struct cnxn_state;

struct cb_ctxt
{
  cnxn_state *state;
  using cb_t = void (*)(cnxn_state *ctxt, ::status_t stat, std::uint64_t, std::size_t len, void *);
  static void cb(void *ctxt, ::status_t stat, std::uint64_t flags, std::size_t len, void *err);
  cb_t cb_call;
  explicit cb_ctxt(cnxn_state *, cb_t);
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

struct cnxn_state
{
  cb_ctxt send_ctxt;
  cb_ctxt recv0_ctxt;
  cb_ctxt recv1_ctxt;
  Component::IFabric_active_endpoint_comm &_comm;
  std::size_t msg_size;
  std::vector<buffer_state> br;
  buffer_state bt;
  unsigned iterations_left;
  explicit cnxn_state(
    cb_ctxt::cb_t send_cb
    , cb_ctxt::cb_t recv0_cb
    , cb_ctxt::cb_t recv1_cb
    , Component::IFabric_active_endpoint_comm &comm_
    , std::size_t buffer_size_
    , std::uint64_t remote_key_
    , unsigned iteration_count_
    , std::size_t msg_size_
  );
  cnxn_state(const cnxn_state &) = delete;
  cnxn_state(cnxn_state &&cs_);
};

#endif
