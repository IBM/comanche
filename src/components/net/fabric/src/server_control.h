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

#ifndef _SERVER_CONTROL_H_
#define _SERVER_CONTROL_H_

#include "fabric_types.h" /* addr_ep_t */
#include "fd_pair.h"

#include <cstddef> /* uint16_t */
#include <map>
#include <memory> /* shared_ptr */
#include <mutex>
#include <queue>
#include <thread>

struct fid_fabric;
struct fid_eq;
class Fabric_connection;
class Fd_socket;

class Pending_cnxns
{
public:
  using cnxn_t = std::shared_ptr<Fabric_connection>;
private:
  std::mutex _m; /* protects _q */
  using guard = std::unique_lock<std::mutex>;
  std::queue<cnxn_t> _q;
public:
  Pending_cnxns();
  void push(cnxn_t c);
  cnxn_t remove();
};

/*
 */

class Server_control
{
  using cnxn_t = std::shared_ptr<Fabric_connection>;
  Pending_cnxns _pending;

  using open_t = std::map<Fabric_connection *, cnxn_t>;
  open_t _open;
  Fd_pair _end;
  std::thread _th;

  static Fd_socket make_listener(std::uint16_t port);
  static void listen(Fd_socket &&listen_fd, int end_fd, fid_fabric &fabric, fid_eq &eq, fabric_types::addr_ep_t name, Pending_cnxns &pend);
public:
  Server_control(fid_fabric &fabric, fid_eq &eq, fabric_types::addr_ep_t name, std::uint16_t port);
  ~Server_control();
  Fabric_connection * get_new_connection();
  std::vector<Fabric_connection *> connections();
  void close_connection(Fabric_connection * cnxn);
};

#endif
