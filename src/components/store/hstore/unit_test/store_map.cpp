#include "store_map.h"

const std::string store_map::impl_default = "hstore";
const store_map::impl_map_t store_map::impl_map = {
  { impl_default, { "hstore", Component::hstore_factory } }
  , { "pmstore", { "pmstore", Component::pmstore_factory } }
  , { "nvmestore", { "nvmestore", Component::nvmestore_factory } }
};

namespace
{
  static auto store_env = ::getenv("STORE");
  static auto store_loc = ::getenv("STORE_LOCATION");
}

const store_map::impl_map_t::const_iterator store_map::impl_env_it =
  store_env
  ? impl_map.find(store_env)
  : impl_map.end()
  ;
const store_map::impl_spec *const store_map::impl =
  &( impl_env_it == impl_map.end()
  ? impl_map.find(impl_default)
  : impl_env_it)->second
  ;
const std::string store_map::location =
  store_loc
  ? store_loc
  : "[ { \"numa_node\": 0, \"path\" : \"/dev/dax0.1\", \"addr\" : " + std::to_string(0x9000000000) + " } ]"
  ;
