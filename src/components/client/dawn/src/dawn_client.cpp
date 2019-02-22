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

IKVStore::pool_t Dawn_client::create_pool(const std::string& path,
                                          const std::string& name,
                                          const size_t       size,
                                          unsigned int       flags,
                                          uint64_t           expected_obj_count)
{
  return _connection->create_pool(path, name, size, flags, expected_obj_count);
}

IKVStore::pool_t Dawn_client::open_pool(const std::string& path,
                                        const std::string& name,
                                        unsigned int       flags)
{
  return _connection->open_pool(path, name, flags);
}

status_t Dawn_client::close_pool(const IKVStore::pool_t pool)
{
  assert(pool);
  return _connection->close_pool(pool);
}

void Dawn_client::delete_pool(const IKVStore::pool_t pool)
{
  assert(pool);
  _connection->delete_pool(pool);
}

status_t Dawn_client::put(const IKVStore::pool_t pool,
                          const std::string&     key,
                          const void*            value,
                          const size_t           value_len)
{
  return _connection->put(pool, key, value, value_len);
}

status_t Dawn_client::put_direct(const pool_t       pool,
                                 const std::string& key,
                                 const void*        value,
                                 const size_t       value_len,
                                 memory_handle_t    handle)
{
  return _connection->put_direct(pool, key, value, value_len, handle);
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

Component::IKVStore::memory_handle_t Dawn_client::register_direct_memory(
    void*  vaddr,
    size_t len)
{
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

size_t Dawn_client::count(const IKVStore::pool_t pool) { return 0; }

void Dawn_client::free_memory(void * p)
{
  ::free(p);
} 

void Dawn_client::debug(const IKVStore::pool_t pool, unsigned cmd, uint64_t arg)
{
}

/* IDawn specific methods */
IKVStore::pool_t Dawn_client::create_pool(const std::string& pool_name,
                                          const size_t size,
                                          unsigned int flags,
                                          uint64_t expected_obj_count) 
{
  return 0;
}

IKVStore::pool_t Dawn_client::open_pool(const std::string& pool_name,
                                       unsigned int flags) 
{
  return 0;
}

std::string Dawn_client::find(const std::string& key_expression,
                              IKVIndex::offset_t begin_position,
                              IKVIndex::find_t find_type,
                              IKVIndex::offset_t& out_end_position) {
  return std::string("");
}

/**
 * Factory entry point
 *
 */
extern "C" void* factory_createInstance(Component::uuid_t& component_id)
{
  if (component_id == Dawn_client_factory::component_id()) {
    auto fact = new Dawn_client_factory();
    fact->add_ref();
    return static_cast<void*>(fact);
  }
  else
    return NULL;
}

#undef RESET_STATE
