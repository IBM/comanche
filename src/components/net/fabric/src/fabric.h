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

#ifndef _FABRIC_
#define _FABRIC_

#include <api/fabric_itf.h>

#include <component/base.h> /* DECLARE_VERSION, DECLARE_COMPONENT_UUID */

#include <cstdint> /* uint64_t */
#include <memory> /* shared_ptr */

struct fi_info;
struct fid_fabric;
struct fid_domain;
struct fid_pep;
struct fid_ep;

class Fabric_connection
  : public Component::IFabric_connection
{
  std::shared_ptr<fid_ep> _ep;
public:
  Fabric_connection(std::shared_ptr<fid_ep> ep, const void *addr, const void *param, size_t paramlen);
  ~Fabric_connection(); /* Note: need to notify the polling thread that this connection is going away, */

  memory_region_t register_memory(const void * contig_addr, size_t size, int flags) override;

  void deregister_memory(const memory_region_t memory_region) override;

  context_t post_send(const std::vector<struct iovec>& buffers) override;

  context_t post_recv(const std::vector<struct iovec>& buffers) override;

  void post_read(
    const std::vector<struct iovec>& buffers,
    uint64_t remote_addr,
    uint64_t key,
    context_t& out_context) override;

  void post_write(
    const std::vector<struct iovec>& buffers,
    uint64_t remote_addr,
    uint64_t key,
    context_t& out_context) override;

  void inject_send(const std::vector<struct iovec>& buffers) override;
  
  int poll_events(std::function<void(context_t, status_t, void*)> completion_callback) override;

  context_t wait_for_next_completion(unsigned polls_limit) override;

  void unblock_completions() override;

  std::string get_peer_addr() override;

  std::string get_local_addr() override;

};


class Fabric_endpoint
  : public Component::IFabric_endpoint
{
  /* A passive edcwndpoint does not need a domain, but an active ep does.
   * Since this class tries to be either, it needs a domain.
   */
  std::shared_ptr<fi_info> _info; 
  std::shared_ptr<fid_pep> _pep; 
  std::shared_ptr<fid_domain> _domain; 
  std::shared_ptr<fid_ep> _aep;
public:
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x8b93a5ae,0xcf34,0x4aff,0x8321,0x19,0x08,0x21,0xa9,0x9f,0xd3);

  Fabric_endpoint(std::shared_ptr<fid_domain> domain_, const fi_info &info);
  void *query_interface(Component::uuid_t& itf_uuid) override;
  
  Fabric_connection * connect(const std::string& remote_endpoint) override;

  Fabric_connection * get_new_connections() override;
  
  void close_connection(Component::IFabric_connection * connection) override;

  std::vector<Component::IFabric_connection*> connections() override;
  
  size_t max_message_size() const override;

  std::string get_provider_name() const override;
};

/*
 * Note: Fabric_factory creates a fabric, a passive endpoint, a [resource] domain, and
 * (when requested) some endpoints. In that order.
 */
class Fabric_factory
  : public Component::IFabric_factory
{
  std::shared_ptr<fi_info> _info; 
  std::shared_ptr<fid_fabric> _fabric; 
  std::shared_ptr<fid_domain> _domain; 
public:
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac3a5ae,0xcf34,0x4aff,0x8321,0x19,0x08,0x21,0xa9,0x9f,0xd3);
  void *query_interface(Component::uuid_t& itf_uuid) override;

  Fabric_factory(const std::string& json_configuration);

  Fabric_endpoint * open_endpoint(const std::string& json_configuration) override;
};

#endif
