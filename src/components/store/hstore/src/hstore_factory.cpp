/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#include "hstore.h"

#include "dax_map.h"

#include <common/utils.h>
//#include <rapidjson/schema.h>

#include <cstdlib> /* getenv */
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
  return create(owner, name, "{[]}");
}

/*
 * See dax_map.cpp for the schema for the JSON "dax_map" parameter.
 */
auto hstore_factory::create(
  const std::string &owner
  , const std::string &name
  , const std::string &dax_map
) -> Component::IKVStore *
{
  Component::IKVStore *obj = new hstore(owner, name, std::make_unique<Devdax_manager>(dax_map, bool(std::getenv("DAX_RESET"))));
  obj->add_ref();
  return obj;
}

auto hstore_factory::create(
  unsigned
  , const std::string &owner
  , const std::string &name
  , const std::string &dax_map
  ) -> Component::IKVStore *
{
  return create(owner, name, dax_map);
}
