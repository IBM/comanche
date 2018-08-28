#ifndef _TEST_PINGPONG_SERVER_N_H_
#define _TEST_PINGPONG_SERVER_N_H_

#include <boost/core/noncopyable.hpp>

#include "pingpong_server_client_state.h"
#include "pingpong_stat.h"
#include <cstdint> /* uint64_t */
#include <thread>
#include <vector>

namespace Component
{
  class IFabric_server_factory;
}

/*
 * A Component::IFabric_server_factory
 */
class pingpong_server_n
  : private boost::noncopyable
{
  std::vector<client_state> _cs;
  pingpong_stat _stat;
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
  pingpong_stat time();
};

#endif
