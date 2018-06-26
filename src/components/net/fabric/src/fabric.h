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

#ifndef _FABRIC_H_
#define _FABRIC_H_

#include <api/fabric_itf.h> /* Component::IFabric */
#include "event_producer.h"

#include <rdma/fabric.h> /* fid_t */
#include <rdma/fi_domain.h> /* fi_eq_attr */

#include <map>
#include <memory> /* shared_ptr */
#include <mutex>

struct fi_info;
struct fid_fabric;
struct fid_eq;
struct fid;

/*
 * Note: Fabric is a fabric which can create servers (IFabric_server_factory) and clients (IFabric_op_completer)
 */
class Fabric
  : public Component::IFabric
  , private event_producer
{
  std::shared_ptr<::fi_info> _info;
  std::shared_ptr<::fid_fabric> _fabric;
  /* an event queue, in case the endpoint is connection-oriented */
  ::fi_eq_attr _eq_attr;
  std::shared_ptr<::fid_eq> _eq;
  int _fd;
  /* A limited number of fids use the event queue:
   *  - connection server factories created by open_server_factory will use it to
   *    - receive connection requests FI_CONNREQ)
   *  - connection clients use it to
   *    - be notified when their connection is accepted (FI_CONNECTION) or rejected (some error event)
   *    - be notified when their connection is shut down (FI_SHUTDOWN)
   *
   * These events are not expected:
   *   FI_NOTIFY (libfabric internal)
   *   FI_MR_COMPLETE (asymc MR)
   *   FI_AV_COMPLETE (async AV)
   *   FI_JOIN_COMPLETE (multicast join)
   */
  using eq_dispatch_t = std::map<::fid_t, event_consumer *>;
  /* Need to add active endpoint in passive endpoint callback, so use separate maps and separate locks */
  std::mutex _m_eq_dispatch_pep;
  eq_dispatch_t _eq_dispatch_pep;
  std::mutex _m_eq_dispatch_aep;
  eq_dispatch_t _eq_dispatch_aep;

  /* BEGIN Component::IFabric */
  Component::IFabric_server_factory * open_server_factory(const std::string& json_configuration, std::uint16_t control_port) override;
  Component::IFabric_client * open_client(const std::string& json_configuration, const std::string & remote, std::uint16_t control_port) override;
  Component::IFabric_server_grouped_factory * open_server_grouped_factory(const std::string& json_configuration, std::uint16_t control_port) override;
  Component::IFabric_client_grouped * open_client_grouped(const std::string& json_configuration, const std::string& remote_endpoint, std::uint16_t port) override;
  /* END Component::IFabric */

  /* BEGIN event_producer */
  void register_pep(::fid_t ep, event_consumer &ec) override;
  void register_aep(::fid_t ep, event_consumer &ec) override;
  void deregister_endpoint(::fid_t ep) override;
  void bind(::fid_ep &ep) override;
  void bind(::fid_pep &ep) override;
  int fd() const override;
  void wait_eq() override;
  void read_eq() override;
  /* END event_producer */
  void readerr_eq();

  /* The help text does not say whether attr may be null, but the provider source expects that it is not. */
  std::shared_ptr<::fid_eq> make_fid_eq(::fi_eq_attr &attr, void *context) const;

public:
  explicit Fabric(const std::string& json_configuration);
  int trywait(::fid **fids, std::size_t count) const;
  std::shared_ptr<::fid_domain> make_fid_domain(::fi_info &info, void *context) const;
  std::shared_ptr<::fid_pep> make_fid_pep(::fi_info &info, void *context) const;
};

#endif
