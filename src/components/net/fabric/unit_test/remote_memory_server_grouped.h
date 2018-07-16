#ifndef _TEST_REMOTE_MEMORY_SERVER_GROUPED_H_
#define _TEST_REMOTE_MEMORY_SERVER_GROUPED_H_

#include "remote_memory_accessor.h"
#include <boost/core/noncopyable.hpp>

#include <cstdint> /* uint16_t, unit64_t */
#include <memory> /* make_shared, shared_ptr */
#include <thread>

namespace Component
{
  class IFabric;
  class IFabric_server_grouped_factory;
}

class remote_memory_server_grouped
  : public remote_memory_accessor
  , private boost::noncopyable
{
  std::shared_ptr<Component::IFabric_server_grouped_factory> _ep;
  std::thread _th;

  void listener(Component::IFabric_server_grouped_factory &ep, std::uint64_t remote_key_base);

public:
  remote_memory_server_grouped(
    Component::IFabric &fabric
    , const std::string &fabric_spec
    , std::uint16_t control_port
    , std::uint64_t remote_key_base
  );

  ~remote_memory_server_grouped();
};

#endif
