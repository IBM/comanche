#ifndef _TEST_REMOTE_MEMORY_SERVER_H_
#define _TEST_REMOTE_MEMORY_SERVER_H_

#include "remote_memory_accessor.h"
#include <boost/core/noncopyable.hpp>

#include <cstdint> /* uint16_t */
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
class remote_memory_server
  : public remote_memory_accessor
  , private boost::noncopyable
{
  std::shared_ptr<Component::IFabric_server_factory> _ep;
  std::thread _th;

  void listener(Component::IFabric_server_factory &ep);

  void listener_counted(Component::IFabric_server_factory &ep, unsigned cnxn_count);
public:
  remote_memory_server(Component::IFabric &fabric, const std::string &fabric_spec, std::uint16_t control_port);
  remote_memory_server(Component::IFabric &fabric, const std::string &fabric_spec, std::uint16_t control_port, unsigned cnxn_limit);
  ~remote_memory_server();
};

#endif
