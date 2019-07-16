/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "dawn_client.h"
#include "connection.h"
#include "protocol.h"

#include <api/fabric_itf.h>
#include <city.h>

#include <iostream>
#include <regex>

using namespace Component;

namespace Dawn
{
namespace Global
{
unsigned debug_level = 0;
}
}  // namespace Dawn

Dawn_client::Dawn_client(unsigned           debug_level,
                         const std::string& owner,
                         const std::string& addr_port_str,
                         const std::string& device)
{
  using namespace std;

  Dawn::Global::debug_level = debug_level;

  smatch m;

  try {
    /* e.g. 10.0.0.21:11911 (verbs)
       9.1.75.6:11911:sockets (sockets)
    */
    regex r("([[:digit:]]+[.][[:digit:]]+[.][[:digit:]]+[.][[:digit:]]+)[:]([[:"
            "digit:]]+)(?:[:]([[:alnum:]]+))?");
    regex_search(addr_port_str, m, r);
  }
  catch (...) {
    throw API_exception("invalid parameter");
  }

  const std::string ip_addr = m[1].str();
  char*             end;
  const int         port = (int) strtoul(m[2].str().c_str(), &end, 10);
  const std::string provider =
      m[3].matched ? m[3].str() : "verbs"; /* default provider */

  PMAJOR("Dawn-client protocol session: %p (%s) (%d) (%s)", this,
         ip_addr.c_str(), port, provider.c_str());

  open_transport(device, ip_addr, port, provider);
}

Dawn_client::~Dawn_client() { close_transport(); }

void Dawn_client::open_transport(const std::string& device,
                                 const std::string& ip_addr,
                                 const int          port,
                                 const std::string& provider)
{
  {
    IBase* comp = load_component("libcomanche-fabric.so", net_fabric_factory);
    assert(comp);
    _factory = static_cast<IFabric_factory*>(
        comp->query_interface(IFabric_factory::iid()));
    assert(_factory);

    /* The libfabric 1.6 sockets provider requires a "BASIC" specfication, which
     * is supposedly obsolete after libfabric 1.4.
     */
    const std::string mr_mode =
        provider == "sockets" ? "[ \"FI_MR_BASIC\" ]"
                              : "[ \"FI_MR_LOCAL\", \"FI_MR_VIRT_ADDR\", "
                                "\"FI_MR_ALLOCATED\", \"FI_MR_PROV_KEY\" ]";

    const std::string fabric_spec{"{ \"fabric_attr\" : { \"prov_name\" : \"" +
                                  provider +
                                  "\" },"
                                  " \"domain_attr\" : "
                                  "{ \"mr_mode\" : " +
                                  mr_mode + " , \"name\" : \"" + device +
                                  "\" }"
                                  ","
                                  " \"ep_attr\" : { \"type\" : \"FI_EP_MSG\" }"
                                  "}"};

    _fabric = _factory->make_fabric(fabric_spec);
    const std::string client_spec{"{}"};
    _transport = _fabric->open_client(client_spec, ip_addr, port);
    assert(_transport);
  }

  assert(_transport);
  _connection = new Dawn::Client::Connection_handler(_transport);
  _connection->bootstrap();
}

void Dawn_client::close_transport()
{
  PLOG("Dawn_client: closing fabric transport (%p)", this);

  if (_connection) {
    _connection->shutdown();
  }

  delete _connection;
  delete _transport;
  delete _fabric;
  _factory->release_ref();
  PLOG("Dawn_client: closed fabric transport.");
}

int Dawn_client::thread_safety() const
{
  return IKVStore::THREAD_MODEL_SINGLE_PER_POOL;
}

int Dawn_client::get_capability(Capability cap) const
{
  switch(cap) {
  case Capability::POOL_DELETE_CHECK: return 1;
  case Capability::POOL_THREAD_SAFE: return 1;
  case Capability::RWLOCK_PER_POOL: return 1;
  default: return -1;
  }
}


IKVStore::pool_t Dawn_client::create_pool(const std::string& name,
                                          const size_t       size,
                                          uint32_t           flags,
                                          uint64_t           expected_obj_count)
{
  return _connection->create_pool(name, size, flags, expected_obj_count);
}

IKVStore::pool_t Dawn_client::open_pool(const std::string& name,
                                        uint32_t       flags)
{
  return _connection->open_pool(name, flags);
}

status_t Dawn_client::close_pool(const IKVStore::pool_t pool)
{
  if(!pool) return E_INVAL;
  return _connection->close_pool(pool);
}

status_t Dawn_client::delete_pool(const std::string& name)
{
  return _connection->delete_pool(name);
}

status_t Dawn_client::configure_pool(const IKVStore::pool_t pool,
                                     const std::string& json)
{
  return _connection->configure_pool(pool, json);
}


status_t Dawn_client::put(const IKVStore::pool_t pool,
                          const std::string&     key,
                          const void*            value,
                          const size_t           value_len,
                          uint32_t               flags)
{
  assert(flags < FLAGS_MAX_VALUE);
  return _connection->put(pool, key, value, value_len, flags);
}

status_t Dawn_client::put_direct(const pool_t       pool,
                                 const std::string& key,
                                 const void*        value,
                                 const size_t       value_len,
                                 memory_handle_t    handle,
                                 uint32_t           flags)
{
  return _connection->put_direct(pool, key, value, value_len, handle, flags);
}

status_t Dawn_client::get(const IKVStore::pool_t pool,
                          const std::string&     key,
                          void*&  out_value, /* release with free() */
                          size_t& out_value_len)
{
  return _connection->get(pool, key, out_value, out_value_len);
}

status_t Dawn_client::get_direct(const pool_t       pool,
                                 const std::string& key,
                                 void*              out_value,
                                 size_t&            out_value_len,
                                 memory_handle_t    handle)
{
  return _connection->get_direct(pool, key, out_value, out_value_len, handle);
}

Component::IKVStore::memory_handle_t
Dawn_client::register_direct_memory(void*  vaddr,
                                    size_t len)
{
  madvise(vaddr, len, MADV_DONTFORK);
  // if(madvise(vaddr, len, MADV_DONTFORK) != 0) {
  //   PWRN("Dawn_client::register_direct_memory:: madvise 'don't fork' failed unexpectedly (%p %lu) %s",
  //        vaddr, len, strerro(errno));
  // }

  return _connection->register_direct_memory(vaddr, len);
}

status_t Dawn_client::unregister_direct_memory(IKVStore::memory_handle_t handle)
{
  return _connection->unregister_direct_memory(handle);
}

status_t Dawn_client::erase(const IKVStore::pool_t pool, const std::string& key)
{
  return _connection->erase(pool, key);
}

size_t Dawn_client::count(const IKVStore::pool_t pool)
{
  return _connection->count(pool);
}

status_t Dawn_client::get_attribute(const IKVStore::pool_t pool,
                                    const IKVStore::Attribute attr,
                                    std::vector<uint64_t>& out_attr,
                                    const std::string* key)
{
  return _connection->get_attribute(pool, attr, out_attr, key);
}

status_t Dawn_client::get_statistics(Shard_stats& out_stats)
{
  return _connection->get_statistics(out_stats);
}


status_t Dawn_client::free_memory(void * p)
{
  ::free(p);
  return S_OK;
} 

void Dawn_client::debug(const IKVStore::pool_t pool, unsigned cmd, uint64_t arg)
{
}

status_t Dawn_client::find(const IKVStore::pool_t pool,
                           const std::string& key_expression,
                           const offset_t offset,
                           offset_t& out_matched_offset,
                           std::string& out_matched_key)
{
  return _connection->find(pool, key_expression, offset, out_matched_offset, out_matched_key);
}

status_t Dawn_client::invoke_ado(const IKVStore::pool_t pool,
                                 const std::string& key,
                                 const std::vector<uint8_t>& request,
                                 const uint32_t flags,                              
                                 std::vector<uint8_t>& out_response,
                                 const size_t value_size)
{
  return _connection->invoke_ado(pool, key, request, flags, out_response, value_size);
}


/**
 * Factory entry point
 *
 */
extern "C" void* factory_createInstance(Component::uuid_t& component_id)
{
  if (component_id == Dawn_client_factory::component_id()) {
    PMAJOR("Creating Dawn_client_factory ...");
    auto fact = new Dawn_client_factory();
    fact->add_ref();
    return static_cast<void*>(fact);
  }
  else {
    PWRN("request for bad factory type");
    return NULL;
  }
}

#undef RESET_STATE
