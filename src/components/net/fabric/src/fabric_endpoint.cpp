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

#include "fabric_endpoint.h"

#include "fabric_connection.h"
#include "fabric_factory.h"
#include "fabric_help.h"
#include "fabric_ptr.h"

#include <rdma/fi_eq.h>

#include <cerrno>
#include <string>

/** 
 * Fabric/RDMA-based network component
 * 
 */

/* size, allow fi_eq_write, provider's choice for wait, no affinity, no wait set */
static fi_eq_attr a { 100, FI_WRITE, FI_WAIT_UNSPEC, 0, nullptr }; 

Fabric_endpoint::Fabric_endpoint(Fabric_factory & fabric_, fi_info &info_)
  : _fabric(fabric_)
  , _pep{make_fid_pep(*_fabric.fid(), info_, this)}
  , _max_msg_size{info_.ep_attr->max_msg_size}
  , _open{}
  , _eq{make_fid_eq(*_fabric.fid(), &a, this)}
  , _run{true}
  , _th{&Fabric_endpoint::listen, std::ref(*this)}
{}

Fabric_endpoint::~Fabric_endpoint()
{
  _run = false;
  _th.join();
}

void Fabric_endpoint::handle_notify(const fi_eq_entry &entry, std::size_t len)
{
  not_implemented(__func__);
}

void Fabric_endpoint::handle_connreq(const fi_eq_cm_entry &entry, std::size_t len)
{
  auto info = std::shared_ptr<fi_info>(entry.info, fi_freeinfo);
  not_implemented(__func__);
}

void Fabric_endpoint::handle_connected(const fi_eq_cm_entry &entry, std::size_t len)
{
  auto info = std::shared_ptr<fi_info>(entry.info, fi_freeinfo);
  not_expected(__func__);
}

void Fabric_endpoint::handle_shutdown(const fi_eq_cm_entry &entry, std::size_t len)
{
  auto info = std::shared_ptr<fi_info>(entry.info, fi_freeinfo);
  not_implemented(__func__);
}

void Fabric_endpoint::handle_mr_complete(const fi_eq_entry &entry, std::size_t len)
{
  not_expected(__func__);
}

void Fabric_endpoint::handle_av_complete(const fi_eq_entry &entry, std::size_t len)
{
  not_expected(__func__);
}

void Fabric_endpoint::handle_join_complete(const fi_eq_entry &entry, std::size_t len)
{
  not_expected(__func__);
}

void Fabric_endpoint::handle_unknown_event(std::uint64_t event, void *buf, std::size_t len)
{
  std::cout << "While listening: unexpected event " << event << "\n";
  not_expected(__func__);
}

void Fabric_endpoint::listen()
{
  while ( _run ) {
    uint32_t event;
    std::vector<char> buf;
    auto i = fi_eq_read(&*_eq,  &event, &*buf.begin(), 0, FI_PEEK);
    if ( 0 < i )
    {
      buf.resize(i);
      i = fi_eq_read(&*_eq,  &event, &*buf.begin(), buf.size(), 0);
        /* interpreting events:
         * FI_NOTIFY => fi_eq_entry, and 'data' field contains the uint64_t representation of an ofi_cmap_signal { OFI_CMAP_FREE, OFI_CMAP_EXIT, } (documented where?)
         * FI_CONNREQ, FI_CONNECTED, and FI_SHUTDOWN => fi_eq_cm_entry
         * FI_MR_COMPLETE => fi_eq_entry
         * FI_AV_COMPLETE => fi_eq_entry
         * FI_JOIN_COMPLETE => fi_eq_entry
         * FI_EOVERRUN => ??
         */
      switch ( event )
      {
      case FI_NOTIFY:
        {
          auto e = static_cast<fi_eq_entry *>(static_cast<void *>(&*buf.begin()));
          handle_notify(*e, i);
        }
        break;
      case FI_CONNREQ:
        {
          auto e = static_cast<fi_eq_cm_entry *>(static_cast<void *>(&*buf.begin()));
          handle_connreq(*e, i);
        }
        break;
      case FI_CONNECTED:
        {
          auto e = static_cast<fi_eq_cm_entry *>(static_cast<void *>(&*buf.begin()));
          handle_connected(*e, i);
        }
        break;
      case FI_SHUTDOWN:
        {
          auto e = static_cast<fi_eq_cm_entry *>(static_cast<void *>(&*buf.begin()));
          handle_shutdown(*e, i);
        }
        break;
      case FI_MR_COMPLETE:
        {
          auto e = static_cast<fi_eq_entry *>(static_cast<void *>(&*buf.begin()));
          handle_mr_complete(*e, i);
        }
        break;
      case FI_AV_COMPLETE:
        {
          auto e = static_cast<fi_eq_entry *>(static_cast<void *>(&*buf.begin()));
          handle_av_complete(*e, i);
        }
        break;
      case FI_JOIN_COMPLETE:
        {
          auto e = static_cast<fi_eq_entry *>(static_cast<void *>(&*buf.begin()));
          handle_join_complete(*e, i);
        }
        break;
      default:
          handle_unknown_event(event, &*buf.begin(), i);
        break;
      }
    }
    /* if i == -FI_EAVAIL, check the error queue with fi_eq_readerr. */
    if ( i < 0 )
    {
      switch ( -i )
      {
      case FI_EAVAIL:
        {
          fi_eq_err_entry err;
          auto j = fi_eq_readerr(&*_eq, &err, 0);
          if ( j == sizeof err )
          {
            std::cout << "While listening: remote error " << err.err << "\n";
          }
          else
          {
            /* error peeking at the remote error ... */
            std::cout << "While listening then while reading remote error: local error " << j << "\n";
          }
        }
        break;
      case EAGAIN:
        break;
      default:
        std::cout << "While listening then while peeking: local error " << -i << " " << fi_strerror(-i) << "\n";
        break;
      }
    }
  }
}

/** 
 * Connect to a waiting peer (creates local endpoint)
 * 
 * @param remote_endpoint Remote endpoint designator
 * 
 */
auto Fabric_endpoint::connect(const std::string& remote_endpoint) -> Component::IFabric_connection *
{
  /* should the remote_endpoint parameter include the service (host:service), or should
   * service be a separate parameter? Since we don't know, support neither yet.
   */
  std::string params{};
  auto info = make_fi_info();
  auto it = _open.insert(new Fabric_connection(*_fabric.fid(), *info, remote_endpoint.c_str(), params.c_str(), params.size()));
  assert(it.second);
  return *it.first;
}

void *Fabric_endpoint::query_interface(Component::uuid_t& itf_uuid) {
  return itf_uuid == Component::IFabric_endpoint::iid() ? this : nullptr;
}

Component::IFabric_connection *Fabric_endpoint::get_new_connections()
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
  auto it = _open.find(connection);
  if ( it != _open.end() )
  {
    _open.erase(it);
  }
}

/** 
 * Used to get a vector of active connection belonging to this
 * end point.
 * 
 * @return Vector of new connections
 */
std::vector<Component::IFabric_connection*> Fabric_endpoint::connections()
{
  std::vector<Component::IFabric_connection *> v;
  std::copy(_open.begin(), _open.end(), std::back_inserter(v));
  return v;
}

/** 
 * Get the maximum message size for the provider
 * 
 * @return Max message size in bytes
 */
std::size_t Fabric_endpoint::max_message_size() const
{
  return _max_msg_size;
}

/** 
 * Get provider name
 * 
 * 
 * @return Provider name
 */
std::string Fabric_endpoint::get_provider_name() const
{
  return _fabric.prov_name();
}
