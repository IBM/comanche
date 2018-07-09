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

#ifndef _FABRIC_SERVER_GENERIC_FACTORY_H_
#define _FABRIC_SERVER_GENERIC_FACTORY_H_

#include "event_consumer.h"

#include "delete_copy.h"
#include "event_registration.h"
#include "fd_pair.h"
#include "fd_socket.h"
#include "pending_cnxns.h"
#include "open_cnxns.h"

#include <cstdint> /* uint16_t */
#include <memory> /* shared_ptr */
#include <thread>

struct fi_info;
struct fid_pep;

class Fabric;
class Fabric_memory_control;
class event_producer;

class Fabric_server_generic_factory
  : public event_consumer
{
  ::fi_info &_info;
  Fabric &_fabric;
  std::shared_ptr<::fid_pep> _pep;
  event_registration _event_registration;

  Pending_cnxns _pending;
  Open_cnxns _open;
  /* a write tells the listener thread to exit */
  Fd_pair _end;

  event_producer &_eq;
  std::thread _th;

  static Fd_socket make_listener(std::uint16_t port);
  void listen(std::uint16_t port, int end_fd, ::fid_pep &pep);

  virtual std::shared_ptr<Fabric_memory_control> new_server(Fabric &fabric, event_producer &eq, ::fi_info &info) = 0;
protected:
  ~Fabric_server_generic_factory();
public:
  /* Note: fi_info is not const because we reuse it when constructing the passize endpoint */
  explicit Fabric_server_generic_factory(Fabric &fabric, event_producer &ev_pr, ::fi_info &info, std::uint16_t control_port);
  DELETE_COPY(Fabric_server_generic_factory);
  Fabric_server_generic_factory(Fabric_server_generic_factory &&) noexcept;

  Fabric_memory_control* get_new_connection();

  void close_connection(Fabric_memory_control* connection);

  std::vector<Fabric_memory_control*> connections();

  std::size_t max_message_size() const;

  std::string get_provider_name() const;

  void cb(std::uint32_t event, ::fi_eq_cm_entry &entry) noexcept override;
  void err(::fi_eq_err_entry &entry) noexcept override;
};

#endif
