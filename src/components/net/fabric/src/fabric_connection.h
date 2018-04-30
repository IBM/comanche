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

#ifndef _FABRIC_CONNECTION_H_
#define _FABRIC_CONNECTION_H_

#include <api/fabric_itf.h>

#include <component/base.h> /* DECLARE_VERSION, DECLARE_COMPONENT_UUID */

#include <memory> /* shared_ptr */

struct fi_info;
struct fid_domain;
struct fid_fabric;
struct fid_ep;

#include <memory> /* shared_ptr */

class Fabric_connection
  : public Component::IFabric_connection
{
  std::shared_ptr<fid_domain> _domain;
  std::shared_ptr<fid_ep> _aep;
public:
  Fabric_connection(fid_fabric &fabric, fi_info &info_, const void *addr, const void *param, size_t paramlen);
  ~Fabric_connection(); /* Note: need to notify the polling thread that this connection is going away, */

  memory_region_t register_memory(const void * contig_addr, size_t size, int flags) override;

  void deregister_memory(const memory_region_t memory_region) override;

  context_t post_send(const std::vector<iovec>& buffers) override;

  context_t post_recv(const std::vector<iovec>& buffers) override;

  void post_read(
    const std::vector<iovec>& buffers,
    uint64_t remote_addr,
    uint64_t key,
    context_t& out_context) override;

  void post_write(
    const std::vector<iovec>& buffers,
    uint64_t remote_addr,
    uint64_t key,
    context_t& out_context) override;

  IFabric_communicator *allocate_group() override;

  void inject_send(const std::vector<iovec>& buffers) override;
  
  std::size_t poll_completions(std::function<void(context_t, status_t, void*, IFabric_communicator *)> completion_callback) override;

  std::size_t stalled_completion_count() override;

  context_t wait_for_next_completion(unsigned polls_limit) override;

  void unblock_completions() override;

  std::string get_peer_addr() override;

  std::string get_local_addr() override;

};

#endif
