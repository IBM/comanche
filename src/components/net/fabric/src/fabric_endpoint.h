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

#ifndef _FABRIC_ENDPOINT_H_
#define _FABRIC_ENDPOINT_H_

#include <api/fabric_itf.h>

#include <component/base.h> /* DECLARE_VERSION, DECLARE_COMPONENT_UUID */

#include <memory> /* shared_ptr */
#include <atomic>
#include <set>
#include <thread>

struct fi_info;
struct fi_eq_entry; /* event queue entry */
struct fi_eq_cm_entry; /* event queue entry: connection management */
struct fid_eq;
struct fid_fabric;
struct fid_pep;

class Fabric_factory;

class Fabric_endpoint
  : public Component::IFabric_endpoint
{
  /* A passive endpoint, in libfabric terms.
   * Plus the ability to listen and allocate an active endpoint.
   */
  /* Despite what might be implied onder getinfo, I cannot find a way to query the attributes
   * of a fabric from fid_fabric. Therefore, retain the pointer to the factory, which includes
   * the fabric and (perhaps) the attributes.
   */
  Fabric_factory & _fabric;
  std::shared_ptr<fid_pep> _pep;
  std::size_t _max_msg_size;
  std::set<Component::IFabric_connection *> _open;
  std::shared_ptr<fid_eq> _eq;
  std::thread _th;

  std::atomic<bool> _run;
  void listen();
  void handle_notify(const fi_eq_entry &, std::size_t len);
  void handle_connreq(const fi_eq_cm_entry &, std::size_t len);
  void handle_connected(const fi_eq_cm_entry &, std::size_t len);
  void handle_shutdown(const fi_eq_cm_entry &, std::size_t len);
  void handle_mr_complete(const fi_eq_entry &, std::size_t len);
  void handle_av_complete(const fi_eq_entry &, std::size_t len);
  void handle_join_complete(const fi_eq_entry &, std::size_t len);
  void handle_unknown_event(std::uint64_t event, void *buf, std::size_t len);
public:
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x8b93a5ae,0xcf34,0x4aff,0x8321,0x19,0x08,0x21,0xa9,0x9f,0xd3);

  Fabric_endpoint(Fabric_factory & fabric, fi_info &info);
  ~Fabric_endpoint();

  void *query_interface(Component::uuid_t& itf_uuid) override;
  
  Component::IFabric_connection * connect(const std::string& remote_endpoint) override;

  Component::IFabric_connection * get_new_connections() override;
  
  void close_connection(Component::IFabric_connection * connection) override;

  std::vector<Component::IFabric_connection *> connections() override;
  
  size_t max_message_size() const override;

  std::string get_provider_name() const override;
};

#endif
