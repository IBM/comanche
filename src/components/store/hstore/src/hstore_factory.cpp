#include "hstore.h"

#include <common/utils.h>

#include <string>

using IKVStore = Component::IKVStore;

/**
 * Factory entry point
 *
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  return
    component_id == hstore_factory::component_id()
    ? new ::hstore_factory()
    : nullptr
    ;
}

void * hstore_factory::query_interface(Component::uuid_t& itf_uuid)
{
  return itf_uuid == Component::IKVStore_factory::iid()
     ? static_cast<Component::IKVStore_factory *>(this)
     : nullptr
     ;
}

void hstore_factory::unload()
{
  delete this;
}

auto hstore_factory::create(
  const std::string &owner
  , const std::string &name
) -> Component::IKVStore *
{
  Component::IKVStore *obj = new hstore(owner, name);
  obj->add_ref();
  return obj;
}

auto hstore_factory::create(
  const std::string &owner
  , const std::string &name
  , const std::string &
) -> Component::IKVStore *
{
  return create(owner, name);
}

auto hstore_factory::create(
  unsigned
  , const std::string &owner
  , const std::string &name
  , const std::string &param2
) -> Component::IKVStore *
{
  return create(owner, name, param2);
}
