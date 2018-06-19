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

#ifndef _FABRIC_CONNECTION_SERVER_FACTORY_H_
#define _FABRIC_CONNECTION_SERVER_FACTORY_H_

#include <api/fabric_itf.h>

#include "event_consumer.h"
#include "event_registration.h"

#include "fabric_ptr.h" /* fid_unique_ptr */
#include "fabric_types.h"
#include "fd_pair.h"
#include "fd_socket.h"
#include "pending_cnxns.h"
#include "open_cnxns.h"

#include <cstdint> /* uint16_t */
#include <memory> /* shared_ptr */
#include <map>
#include <thread>

struct fi_info;
struct fid_fabric;
struct fid_eq;
struct fid_pep;

class Fabric;
class event_producer;

class Fabric_connection_server_factory
  : public Component::IFabric_endpoint
  , public event_consumer
{
  ::fi_info &_info;
  Fabric &_fabric;
  std::shared_ptr<::fid_pep> _pep;
  event_registration _event_registration;

  /* The CONNREQ callback uses _info, so at most one connection request may be pending */
  ::fi_info *_conn_info;

  using cnxn_t = std::shared_ptr<Fabric_connection>;
  Pending_cnxns _pending;
  Open_cnxns _open;
  /* a write tells the listener thread to exit */
  Fd_pair _end;

  event_producer &_eq;
  std::thread _th;

  static Fd_socket make_listener(std::uint16_t port);
  void listen(std::uint16_t port, int end_fd, ::fid_pep &pep, fabric_types::addr_ep_t name);
public:
  /* Note: fi_info is not const because we reuse it when constructing the passize endpoint */
  explicit Fabric_connection_server_factory(Fabric &fabric, event_producer &ev_pr, ::fi_info &info, std::uint16_t control_port);
  Fabric_connection_server_factory(const Fabric_connection_server_factory &) = delete;
  Fabric_connection_server_factory& operator=(const Fabric_connection_server_factory &) = delete;
  ~Fabric_connection_server_factory();

  void *query_interface(Component::uuid_t& itf_uuid) override;

  Component::IFabric_connection * get_new_connections() override;

  void close_connection(Component::IFabric_connection * connection) override;

  std::vector<Component::IFabric_connection*> connections() override;

  std::size_t max_message_size() const override;

  std::string get_provider_name() const override;

  void cb(std::uint32_t event, ::fi_eq_cm_entry &entry) noexcept override;
  void err(::fi_eq_err_entry &entry) noexcept override;

  Fabric_connection * get_new_connection();
  void close_connection(Fabric_connection * cnxn);
};

#endif
