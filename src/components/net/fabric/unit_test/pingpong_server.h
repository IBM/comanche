#ifndef _TEST_PINGPONG_SERVER_H_
#define _TEST_PINGPONG_SERVER_H_

#include <boost/core/noncopyable.hpp>

#include <cstdint> /* uint16_ti, uint64_t */
#include <string>
#include <memory> /* shared_ptr */
#include <thread>

namespace Component
{
  class IFabric;
  class IFabric_server_factory;
}

/*
 * A Component::IFabric_server_factory, which will support clients until one
 * of them closes with the "quit" flag set.
 */
class pingpong_server
  : private boost::noncopyable
{
  std::shared_ptr<Component::IFabric_server_factory> _ep;
  std::thread _th;

  void listener(
    Component::IFabric_server_factory &ep
    , std::size_t buffer_size
    , std::uint64_t remote_key
    , unsigned iteration_count
    , std::size_t msg_size
  );
public:
  pingpong_server(
    Component::IFabric &fabric
    , const std::string &fabric_spec
    , std::uint16_t control_port
    , std::size_t buffer_size
    , std::uint64_t remote_key_base
    , unsigned iteration_count
    , std::size_t msg_size
  );
  ~pingpong_server();
  std::size_t max_message_size() const;
};

#endif
