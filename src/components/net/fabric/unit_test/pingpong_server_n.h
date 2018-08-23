#ifndef _TEST_PINGPONG_SERVER_N_H_
#define _TEST_PINGPONG_SERVER_N_H_

#include <boost/core/noncopyable.hpp>

#include "registered_memory.h"
#include "server_connection.h"
#include <sys/uio.h> /* iovec */
#include <chrono> /* high_resolution_clock */
#include <cstdint> /* uint16_ti, uint64_t */
#include <thread>
#include <utility> /* pair */
#include <vector>

namespace Component
{
  class IFabric;
  class IFabric_server_factory;
}

struct client_state;

struct cb_ctxt
{
  client_state *state;
  using cb_t = void (*)(client_state *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *);
  cb_t cb_call;
  explicit cb_ctxt(client_state *, cb_t);
};

struct client_state
{
  cb_ctxt send_ctxt;
  cb_ctxt recv_ctxt;
  server_connection sc;
  registered_memory rm;
  std::vector<::iovec> v;
  std::vector<void *> d;
  std::size_t msg_size;
  unsigned iterations_left;
  explicit client_state(Component::IFabric_server_factory &factory_, std::size_t buffer_size_, std::uint64_t remote_key_, unsigned iteration_count_, std::size_t msg_size_);
  client_state(const client_state &) = delete;
  client_state(client_state &&cs_);
};

/*
 * A Component::IFabric_server_factory
 */
class pingpong_server_n
  : private boost::noncopyable
{
  std::vector<client_state> _cs;
  std::chrono::high_resolution_clock::time_point _start;
  std::chrono::high_resolution_clock::time_point _stop;
  std::thread _th;

  void listener(
    std::size_t msg_size
  );
public:
  pingpong_server_n(
    unsigned client_count
    , Component::IFabric_server_factory &factory
    , std::size_t buffer_size
    , std::uint64_t remote_key_base
    , unsigned iteration_count
    , std::size_t msg_size
  );
  ~pingpong_server_n();
  std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point> time();
};

#endif
