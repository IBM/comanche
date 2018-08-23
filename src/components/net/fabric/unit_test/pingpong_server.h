#ifndef _TEST_PINGPONG_SERVER_H_
#define _TEST_PINGPONG_SERVER_H_

#include <boost/core/noncopyable.hpp>

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
  std::chrono::high_resolution_clock::time_point _start;
  std::chrono::high_resolution_clock::time_point _stop;
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
  std::pair<std::chrono::high_resolution_clock::time_point,std::chrono::high_resolution_clock::time_point> time();
};

#endif
