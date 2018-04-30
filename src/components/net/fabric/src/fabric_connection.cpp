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

#include <rdma/fi_cm.h> /* fi_connect */

/** 
 * Fabric/RDMA-based network component
 * 
 */

Fabric_connection::Fabric_connection(fid_fabric &fabric_, fi_info &info_, const void *addr, const void *param, size_t paramlen)
  : _domain(make_fid_domain(fabric_, info_, this))
  , _aep{make_fid_aep(*_domain, info_, this)}
{
  auto i = fi_connect(&*_aep, addr, param, paramlen);
  if ( i != FI_SUCCESS )
  {
    throw fabric_error(i, __FILE__, __LINE__);
  }
}

Fabric_connection::~Fabric_connection()
{}

/** 
 * Register buffer for RDMA
 * 
 * @param contig_addr Pointer to contiguous region
 * @param size Size of buffer in bytes
 * @param flags Flags e.g., FI_REMOTE_READ|FI_REMOTE_WRITE
 * 
 * @return Memory region handle
 */
auto Fabric_connection::register_memory(const void * contig_addr, size_t size, int flags) -> Component::IFabric_connection::memory_region_t
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
