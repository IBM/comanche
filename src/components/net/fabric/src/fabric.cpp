/*
   Copyright [2018] [IBM Corporation]

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

/* 
 * Authors: 
 * 
 */

#include "fabric.h"

#include <rapidjson/document.h>

/* fi_connect / fi_listen / fi_accept / fi_reject / fi_shutdown : Manage endpoint connection state.
 *      fi_setname / fi_getname / fi_getpeer : Set local, or return local or peer endpoint address.
 *     fi_join / fi_close / fi_mc_addr : Join, leave, or retrieve a multicast address.
 */
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h> /* fi_endpoint */


#include <map>
#include <memory> /* shared_ptr */

/** 
 * Fabric/RDMA-based network component
 * 
 */

/* A simplification; most fabric errors will not be logic errors */
class fabric_error
  : public std::logic_error
{
public:
  fabric_error(int i, int line)
    : std::logic_error{std::string{"fabric_error \""} + fi_strerror(-i) + "\" at " + std::to_string(line)}
  {}
};

class fabric_bad_alloc
  : public std::bad_alloc
{
  std::string _what;
public:
  fabric_bad_alloc(std::string which)
    : std::bad_alloc{}
    , _what{"fabric_bad_alloc " + which}
  {}
  const char *what() const noexcept override { return _what.c_str(); }
};

template <typename T>
int fi_close_generic(T *f)
{
  return fi_close(&f->fid);
}

#include <iostream>

namespace
{

  std::map<std::string, int> cap_fwd = {
#define X(Y) {#Y, (Y)},
#include "fabric_caps.h"
#undef X
  };

  std::shared_ptr<fid_domain> make_fi_domain(fid_fabric &fabric, fi_info &info, void *context)
  {
    fid_domain *f(nullptr);
    int i = fi_domain(&fabric, &info, &f, context);
    if ( i != FI_SUCCESS )
    {
      std::cout << "FABRIC at " << static_cast<void *>(&fabric) << "\n";
      std::cout << "INFO at " << static_cast<void *>(&fabric) << ":" << fi_tostr(&info, FI_TYPE_INFO) << "\n";
      throw fabric_error(i,__LINE__);
    }
    return std::shared_ptr<fid_domain>(f, fi_close_generic<fid_domain>);
  }

  fid_fabric *make_fi_fabric(fi_fabric_attr &attr, void *context)
  {
    fid_fabric *f(nullptr);
    int i = fi_fabric(&attr, &f, context);
    if ( i != FI_SUCCESS )
    {
      throw fabric_error(i,__LINE__);
    }
    return f;
  }

  std::shared_ptr<fi_info> make_fi_info(
    int version
    , const char *node
    , const char *service
    , uint64_t flags
    , const struct fi_info *hints
  )
  {
    fi_info *f;
    int i = fi_getinfo(version, node, service, flags, hints, &f);
    if ( i != FI_SUCCESS )
    {
      throw fabric_error(i,__LINE__);
    }
    return std::shared_ptr<fi_info>(f,fi_freeinfo);
  }

  std::shared_ptr<fi_info> make_fi_info(const std::string &, std::uint64_t caps)
  {
    /* the preferred provider string is ignored for now. */
    return make_fi_info(fi_version(), nullptr, nullptr, caps|(FI_ASYNC_IOV)|FI_FORMAT_UNSPEC, nullptr);
  }

  std::shared_ptr<fi_info> make_fi_info(const std::string& json_configuration)
  {
    rapidjson::Document jdoc;
    jdoc.Parse(json_configuration.c_str());

    auto provider = jdoc.FindMember("preferred_provider");
    auto provider_str = std::string(provider != jdoc.MemberEnd() && provider->value.IsString() ? provider->value.GetString() : "");

    std::uint64_t caps_int{0U};
    auto caps = jdoc.FindMember("caps");
    if ( caps != jdoc.MemberEnd() && caps->value.IsArray() )
    {
      for ( auto cap = caps->value.Begin(); cap != caps->value.End(); ++cap )
      {
        if ( cap->IsString() )
        {
          auto cap_int_i = cap_fwd.find(cap->GetString());
          if ( cap_int_i != cap_fwd.end() )
          {
            caps_int |= cap_int_i->second;
          }
        }
      }
    }
    return make_fi_info(provider_str, caps_int);
  }

  std::shared_ptr<fid_ep> make_fid_ep(fid_domain &domain, fi_info &info, void *context)
  {
    fid_ep *e;
    int i = fi_endpoint(&domain, &info, &e, context);
    static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
    if ( i != FI_SUCCESS )
    {
      throw fabric_error(i,__LINE__);
    }
    return std::shared_ptr<fid_ep>(e, fi_close_generic<fid_ep>);
  }

  fid_pep *make_fid_pep(fid_fabric &fabric, fi_info &info, void *context)
  {
    fid_pep *e;
    int i = fi_passive_ep(&fabric, &info, &e, context);
    static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
    if ( i != FI_SUCCESS )
    {
      throw fabric_error(i,__LINE__);
    }
    return e;
  }

  std::shared_ptr<fi_info> make_fi_infodup(const fi_info *info)
  {
    auto f = std::shared_ptr<fi_info>(fi_dupinfo(info), fi_freeinfo);
    if ( ! f )
    {
      throw fabric_bad_alloc("fi_info");
    }
    return f;
  }
}

/** 
 * Connect to a waiting peer (creates local endpoint)
 * 
 * @param remote_endpoint Remote endpoint designator
 * 
 */
auto Fabric_endpoint::connect(const std::string& remote_endpoint) -> Fabric_connection *
{
  /* should the remote_endpoint parameter include the service (host:service), or should
   * service be a separate parameter? Since we don't know, support neither yet.
   */
  if ( ! _aep )
  {
    _aep = make_fid_ep(*_domain, *_info, this);
    _pep.reset();
  }
  std::string params{};
  return new Fabric_connection(_aep, remote_endpoint.c_str(), params.c_str(), params.size());
}

Fabric_connection::Fabric_connection(std::shared_ptr<fid_ep> ep_, const void *addr, const void *param, size_t paramlen)
  : _ep{ep_}
{
  auto i = fi_connect(&*_ep, addr, param, paramlen);
  if ( i != FI_SUCCESS )
  {
    throw fabric_error(i,__LINE__);
  }
}

Fabric_connection::~Fabric_connection()
{}

namespace
{
  [[noreturn]] void not_implemented(const std::string &who)
  {
    throw std::runtime_error{who + " not_implemented"};
  }
}

  /** 
   * Register buffer for RDMA
   * 
   * @param contig_addr Pointer to contiguous region
   * @param size Size of buffer in bytes
   * @param flags Flags e.g., FI_REMOTE_READ|FI_REMOTE_WRITE
   * 
   * @return Memory region handle
   */
auto Fabric_connection::register_memory(const void * contig_addr, size_t size, int flags) -> Component::IFabric_connection::memory_region_t
{
  not_implemented(__func__);
}

  /** 
   * De-register memory region
   * 
   * @param memory_region 
   */
void Fabric_connection::deregister_memory(const memory_region_t memory_region)
{
  not_implemented(__func__);
}

  /** 
   * Asynchronously post a buffer to the connection
   * 
   * @param connection Connection to send on
   * @param buffers Buffer vector (containing regions should be registered)
   * 
   * @return Work (context) identifier
   */
auto Fabric_connection::post_send(
  const std::vector<struct iovec>& buffers) -> context_t
{
  not_implemented(__func__);
}

  /** 
   * Asynchronously post a buffer to receive data
   * 
   * @param connection Connection to post to
   * @param buffers Buffer vector (containing regions should be registered)
   * 
   * @return Work (context) identifier
   */
auto Fabric_connection::post_recv(
  const std::vector<struct iovec>& buffers) -> context_t
{
  not_implemented(__func__);
}

  /** 
   * Post RDMA read operation
   * 
   * @param connection Connection to read on
   * @param buffers Destination buffer vector
   * @param remote_addr Remote address
   * @param key Key for remote address
   * @param out_context 
   * 
   */
void Fabric_connection::post_read(
  const std::vector<struct iovec>& buffers,
  uint64_t remote_addr,
  uint64_t key,
  context_t& out_context)
{
  not_implemented(__func__);
}

  /** 
   * Post RDMA write operation
   * 
   * @param connection Connection to write to
   * @param buffers Source buffer vector
   * @param remote_addr Remote address
   * @param key Key for remote address
   * @param out_context 
   * 
   */
void Fabric_connection::post_write(
  const std::vector<struct iovec>& buffers,
  uint64_t remote_addr,
  uint64_t key,
  context_t& out_context)
{
  not_implemented(__func__);
}

  /** 
   * Send message without completion
   * 
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
void Fabric_connection::inject_send(const std::vector<struct iovec>& buffers)
{
  not_implemented(__func__);
}
  
  /** 
   * Poll events (e.g., completions)
   * 
   * @param completion_callback (context_t, status_t status, void* error_data)
   * 
   * @return Number of completions processed
   */
int Fabric_connection::poll_events(std::function<void(context_t, status_t, void*)> completion_callback)
{
  not_implemented(__func__);
}

/** 
 * Block and wait for next completion.
 * 
 * @param polls_limit Maximum number of polls (throws exception on exceeding limit)
 * 
 * @return Next completion context
 */
auto Fabric_connection::wait_for_next_completion(unsigned polls_limit) -> context_t
{
  not_implemented(__func__);
}

  /** 
   * Unblock any threads waiting on completions
   * 
   */
void Fabric_connection::unblock_completions()
{
  not_implemented(__func__);
}

  /** 
   * Get address of connected peer (taken from fi_getpeer during
   * connection instantiation).
   * 
   * 
   * @return Peer endpoint address
   */
std::string Fabric_connection::get_peer_addr()
{
  not_implemented(__func__);
}

  /** 
   * Get local address of connection (taken from fi_getname during
   * connection instantiation).
   * 
   * 
   * @return Local endpoint address
   */

std::string Fabric_connection::get_local_addr()
{
  not_implemented(__func__);
}

/** 
 * Open a fabric provider instance
 * 
 * @param json_configuration Configuration string in JSON
 * form. e.g. {
 *   "caps":["FI_MSG","FI_RMA"],
 *   "preferred_provider" : "verbs",
 *   "bootstrap_addr":"10.0.0.1:9999" }
 * @return 
 *
 * caps: 
 * preferred_provider: same format as struct fi_fabric_attr::prov_name
 */

#include <rapidjson/document.h>

#include <cstdint>

Fabric_factory::Fabric_factory(const std::string& json_configuration)
  : _info(make_fi_info(json_configuration)) 
  , _fabric(make_fi_fabric(*_info->fabric_attr, this))
  , _domain(make_fi_domain(*_fabric, *_info, this))
{
}

void *Fabric_factory::query_interface(Component::uuid_t& itf_uuid) {
  return itf_uuid == IFabric_factory::iid() ? this : nullptr;
}

Fabric_endpoint * Fabric_factory::open_endpoint(const std::string& json_configuration)
{
  rapidjson::Document jdoc;
  jdoc.Parse(json_configuration.c_str());

  auto provider = jdoc.FindMember("preferred_provider");
  auto provider_str = std::string(provider != jdoc.MemberEnd() && provider->value.IsString() ? provider->value.GetString() : "");

  std::uint64_t caps_int{0U};
  auto caps = jdoc.FindMember("caps");
  if ( caps != jdoc.MemberEnd() && caps->value.IsArray() )
  {
    for ( auto cap = caps->value.Begin(); cap != caps->value.End(); ++cap )
    {
      if ( cap->IsString() )
      {
        auto cap_int_i = cap_fwd.find(cap->GetString());
        if ( cap_int_i != cap_fwd.end() )
        {
          caps_int |= cap_int_i->second;
        }
      }
    }
  }

  return new Fabric_endpoint(_domain, *_info);
}

Fabric_endpoint::Fabric_endpoint(std::shared_ptr<fid_domain> domain_, const fi_info &info)
  : _info(make_fi_infodup(&info))
  , _domain(domain_)
  , _pep{make_fid_pep(*_info->fabric_attr->fabric, *_info, this), fi_close_generic<fid_pep>}
  , _aep{}
{}


void *Fabric_endpoint::query_interface(Component::uuid_t& itf_uuid) {
  return itf_uuid == Component::IFabric_endpoint::iid() ? this : nullptr;
}

Fabric_connection *Fabric_endpoint::get_new_connections()
{
  not_implemented(__func__);
}

  /** 
   * Close connection and release any associated resources
   * 
   * @param connection 
   */
void Fabric_endpoint::close_connection(Component::IFabric_connection * connection)
{
  not_implemented(__func__);
}

  /** 
   * Used to get a vector of active connection belonging to this
   * end point.
   * 
   * @return Vector of new connections
   */
std::vector<Component::IFabric_connection*> Fabric_endpoint::connections()
{
  not_implemented(__func__);
}

  /** 
   * Get the maximum message size for the provider
   * 
   * @return Max message size in bytes
   */
std::size_t Fabric_endpoint::max_message_size() const
{
  not_implemented(__func__);
}

  /** 
   * Get provider name
   * 
   * 
   * @return Provider name
   */
std::string Fabric_endpoint::get_provider_name() const
{
  not_implemented(__func__);
}

#include <iostream>

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  return component_id == Fabric_factory::component_id() ? new Fabric_factory("{}") : nullptr;
}

