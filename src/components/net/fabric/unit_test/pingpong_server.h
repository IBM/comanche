#ifndef _TEST_PINGPONG_SERVER_H_
#define _TEST_PINGPONG_SERVER_H_

#include <boost/core/noncopyable.hpp>

#include "pingpong_stat.h"
#include "server_connection.h"
#include <chrono> /* high_resolution_clock */
#include <cstdint> /* uint16_ti, uint64_t */
#include <string>
#include <memory> /* shared_ptr */
#include <thread>
#include <utility> /* pair */

namespace Component
{
  class IFabric;
  class IFabric_server_factory;
}

/*
 * A Component::IFabric_server_factory
 */
class pingpong_server
  : private boost::noncopyable
{
  server_connection _sc;
  pingpong_stat _stat;
  std::thread _th;

  void listener(
    std::size_t buffer_size
    , std::uint64_t remote_key
    , unsigned iteration_count
    , std::size_t msg_size
  );
public:
  pingpong_server(
    Component::IFabric_server_factory &factory
    , std::size_t buffer_size
    , std::uint64_t remote_key_base
    , unsigned iteration_count
    , std::size_t msg_size
  );
  ~pingpong_server();
  pingpong_stat time();
};

#endif
