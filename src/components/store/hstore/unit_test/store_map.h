#ifndef _DAWN_HSTORE_TEST_STORE_MAP_H_
#define _DAWN_HSTORE_TEST_STORE_MAP_H_

#include <api/components.h>

#include <string>
#include <map>

// The fixture for testing class Foo.
class store_map
{
public:
  struct impl_spec
  {
    std::string       name;
    Component::uuid_t factory_id;
  };

  using impl_map_t = std::map<std::string, impl_spec>;
private:
  static const std::string impl_default;
  static const impl_map_t impl_map;
  static const impl_map_t::const_iterator impl_env_it;
  static const impl_spec pmstore_impl;
public:
  static const impl_spec *const impl;
  static const std::string location;
};

#endif
