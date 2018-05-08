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

#include "fabric_connection.h"

#include "fabric_error.h"
#include "fabric_help.h"

#include <rdma/fabric.h> /* FI_TYPE_xxx */
#include <rdma/fi_cm.h> /* fi_connect */

/** 
 * Fabric/RDMA-based network component
 * 
 */


/* Client-side constructor */
Fabric_connection::Fabric_connection(fid_fabric &fabric_, const fi_info &info_, Fd_control &&control_, bool is_client_
  )
  : _descr( is_client_ ? "client " : "server " )
  , _control(std::move(control_))
  /* client needs the address format, before it creates the domain */
  , _addr_format(( is_client_ ? _control.recv_format() : format_ep_t(info_.addr_format)))
  , _domain_info(make_fi_infodup(info_, "domain construction"))
  , _domain(((_domain_info->addr_format = std::get<0>(_addr_format)), make_fid_domain(fabric_, *_domain_info, this)))
  , _tx(*_domain, FI_SEND | FI_WRITE | FI_REMOTE_READ, tx_key
      , 0U // was mr_flag, flags for fid_mr_regs ( FI_RMA_EVENT , FI_RMA_PMEM )
    )
  , _rx(*_domain, FI_RECV | FI_READ | FI_REMOTE_WRITE, rx_key
      , 0U // was mr_flags, flags for fid_mr_reg ( FI_RMA_EVENT , FI_RMA_PMEM )
    )
  , _av_attr{}
  , _av(make_fid_av(*_domain, _av_attr, this))
  , _ep_info(make_fi_infodup(*_domain_info, "endpoint construction"))
  , _ep(make_fid_aep(*_domain, *_ep_info, this))
  , _eq_attr{}
  , _eq{}
{
  /* NOTE: the various tests for type (FI_EP_MSG) and mode (_is_client)
   * should move to derived classses.
   */
  if ( _ep_info->ep_attr->type == FI_EP_MSG )
  {
    _eq_attr.size = 10;
    _eq_attr.wait_obj = FI_WAIT_NONE; 
    _eq = make_fid_eq(fabric_, &_eq_attr, this);
    CHECKZ(fi_ep_bind(&*_ep, &_eq->fid, 0));
  }

  if ( _av )
  {
    CHECKZ(fi_ep_bind(&*_ep, &_av->fid, 0));
  }
  CHECKZ(fi_ep_bind(&*_ep, _tx.cq(), FI_TRANSMIT));
  CHECKZ(fi_ep_bind(&*_ep, _rx.cq(), FI_RECV));

  CHECKZ(fi_enable(&*_ep));

  /* Give the endpoint a receive buffer */
  post(
    fi_recv, get_rx_comp, _rx.seq()
    , this /* context, to get_rx_comp */
    , "receive " __FILE__ /* string identifier for reporting */
    /* parameters to fi_recv: */
    , &*_ep /* context */
    , static_cast<void *>(&*this->_rx.buffer().begin())
    , this->_rx.buffer().size()
    , this->_rx.mr_desc()
    , FI_ADDR_UNSPEC /* src_addr */
    , static_cast<void *>(this)
  );

  if ( is_client_ )
  {
    auto param = nullptr;
    std::size_t paramlen = 0;
    auto remote_addr = _control.recv_name();
    if ( _ep_info->ep_attr->type == FI_EP_MSG )
    {
      try
      {
        CHECKZ(fi_connect(&*_ep, &*std::get<0>(remote_addr).begin(), param, paramlen));
      }
      catch ( const fabric_error &e )
      {
        throw e.add(fi_tostr(&*_ep_info, FI_TYPE_INFO));
      }
    }
  }
}

Fabric_connection::~Fabric_connection()
{
  /* "the flags parameter is reserved and must be 0" */
  fi_shutdown(&*_ep, 0);
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
auto Fabric_connection::register_memory(const void * contig_addr, size_t size, std::uint64_t key, int flags) -> Component::IFabric_connection::memory_region_t
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
   * Poll completions (e.g., completions)
   * 
   * @param completion_callback (context_t, status_t status, void* error_data)
   * 
   * @return Number of completions processed
   */
std::size_t Fabric_connection::poll_completions(std::function<void(context_t, status_t, void*, IFabric_communicator *)> completion_callback)
{
  not_implemented(__func__);
}

std::size_t Fabric_connection::stalled_completion_count()
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
#pragma GCC diagnostic pop

auto Fabric_connection::allocate_group() -> IFabric_communicator *
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

auto Fabric_connection::get_name() const -> addr_ep_t
{
  auto it = static_cast<const char *>(_ep_info->src_addr);
  return addr_ep_t(
     std::vector<char>(it, it+_ep_info->src_addrlen)
  );
}

void Fabric_connection::get_rx_comp(void *ctx_, uint64_t total_)
{
  auto ctx = static_cast<Fabric_connection *>(ctx_);
  if ( ! ctx )
  {
    throw std::logic_error("call to rx_comp with no rx resources");
  }
  auto &rx = ctx->_rx;
  return rx.get_cq_comp(total_);
}
