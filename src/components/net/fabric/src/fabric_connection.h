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

#include "fd_control.h"
#include "endpoint_resources_simplex.h"

#include <component/base.h> /* DECLARE_VERSION, DECLARE_COMPONENT_UUID */

#include <memory> /* shared_ptr */
#include <tuple>

struct fi_info;
struct fid_domain;
struct fid_fabric;
struct fid_ep;

class Fabric_connection
  : public Component::IFabric_connection
{
  static constexpr auto tx_key = std::uint64_t(0x0123456789abcdef);
  static constexpr auto rx_key = std::uint64_t(tx_key + 1U);
  using format_ep_t = std::tuple<std::uint32_t>;
  using addr_ep_t = std::tuple<std::vector<char>>;
  std::string _descr;
  Fd_control _control;
  format_ep_t _addr_format; /* HACK: necessary (addr_format, at least) on client side to construct the domain. Don't know whether server side will need it. */
  std::shared_ptr<fi_info> _domain_info;
  std::shared_ptr<fid_domain> _domain;
  endpoint_resources_simplex _tx;
  endpoint_resources_simplex _rx;
  fi_av_attr _av_attr;
  fid_unique_ptr<fid_av> _av;
  std::shared_ptr<fi_info> _ep_info;
  std::shared_ptr<fid_ep> _ep;
  /* an event queue, use only if the endpoint turns out to be type FI_EP_MSG */
  fi_eq_attr _eq_attr;
  std::shared_ptr<fid_eq> _eq;

  static void get_rx_comp(void *ctx, uint64_t limit);
protected:
  static constexpr std::uint64_t control_port = 47591;
public:
  Fabric_connection(fid_fabric &fabric_, const fi_info &info_, Fd_control &&control_, bool is_client);

  ~Fabric_connection(); /* Note: need to notify the polling thread that this connection is going away, */

  const Fd_control &control() const { return _control; }

  memory_region_t register_memory(const void * contig_addr, size_t size, uint64_t key, int flags) override;

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

  addr_ep_t get_name() const;
};

#endif
