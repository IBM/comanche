/*
  Copyright [2017] [IBM Corporation]

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
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __CORE_UIPC_H__
#define __CORE_UIPC_H__

#include <common/mpmc_bounded_queue.h>
#include <sys/stat.h>
#include <mutex>
#include <string>
#include <vector>

/* Expose as C because we want other programming languages to
   interface onto this API */

extern "C" {
typedef void* channel_t;

/**
 * Channel is bi-directional, user-level, lock-free exchange of
 * fixed sized messages (zero-copy).  Receiving is polling-based (it
 * can be configured as sleeping. It does not define the message
 * protocol which can be Protobuf etc.  Channel is a lock-free FIFO
 * (MPMC) in shared memory for passing pointers together with a slab
 * allocator (also lock-free and thread-safe across both sides) for
 * fixed sized messages.  Both sides map to the same virtual
 * address; this is negogiated. 4M exchanges per second is typical.
 *
 */

/**
 * Create a channel and wait for client to connect
 *
 * @param path_name Unique name (e.g., /tmp/myChannel)
 * @param message_size Message size in bytes.
 * @param queue_size Max elements in FIFO
 *
 * @return Handle to channel or NULL on failure
 */
channel_t uipc_create_channel(const char* path_name, size_t message_size,
                              size_t queue_size);

/**
 * Connect to a channel
 *
 * @param path_name Name of channel (e.g., /tmp/myChannel)
 * @param
 *
 * @return Handle to channel or NULL on failure
 */
channel_t uipc_connect_channel(const char* path_name);

/**
 * Close a channel
 *
 * @param channel Channel handle
 *
 * @return S_OK or E_INVAL
 */
status_t uipc_close_channel(channel_t channel);

/**
 * Allocate a fixed size message
 *
 * @param channel Associated channel
 *
 * @return Pointer to message in shared memory or NULL on failure
 */
void* uipc_alloc_message(channel_t channel);

/**
 * Free message
 *
 * @param channel Associated channel
 * @param message Message
 *
 * @return
 */
status_t uipc_free_message(channel_t channel, void* message);

/**
 * Send a message
 *
 * @param channel Channel handle
 * @param data Pointer to data in channel memory
 *
 * @return S_OK or E_FULL
 */
status_t uipc_send(channel_t channel, void* data);

/**
 * Recv a message
 *
 * @param channel Channel handle
 * @param data_out Pointer too data popped off FIFO
 *
 * @return S_OK or E_EMPTY
 */
status_t uipc_recv(channel_t channel, void*& data_out);
}

namespace Core
{
namespace UIPC
{
class Shared_memory {
 private:
  static constexpr bool option_DEBUG = false;

 public:
  Shared_memory(std::string name, size_t n_pages); /*< initiator constructor */
  Shared_memory(std::string name);                 /*< target constructor */

  virtual ~Shared_memory() noexcept(false);

  void* get_addr(size_t offset = 0);

  size_t get_size_in_pages() const { return _size_in_pages; }
  size_t get_size() const { return _size_in_pages * PAGE_SIZE; }

 private:
  void* negotiate_addr_create(std::string path_name, size_t size_in_bytes);

  void* negotiate_addr_connect(std::string path_name,
                               size_t* size_in_bytes_out);

  void open_shared_memory(std::string path_name, bool master);

 private:
  bool _master;
  std::vector<std::string> _fifo_names;
  std::string _name;
  void* _vaddr;
  size_t _size_in_pages;
};

class Channel {
 private:
  static constexpr bool option_DEBUG = false;
  typedef Common::Mpmc_bounded_lfq<void*> queue_t;

 public:
  /**
   * Master-side constructor
   *
   * @param name Name of channel
   * @param message_size Size of messages in bytes
   * @param queue_size Max elements in FIFO
   */
  Channel(std::string name, size_t message_size, size_t queue_size);

  /**
   * Slave-side constructor
   *
   * @param name Name of channel
   * @param message_size Size of messages in bytes
   * @param queue_size Max elements in FIFO
   */
  Channel(std::string name);

  /**
   * Destructor
   *
   *
   * @return
   */
  virtual ~Channel();

  /**
   * Post message onto channel
   *
   * @param msg Message pointer
   *
   * @return S_OK or E_FULL
   */
  status_t send(void* msg);

  /**
   * Receive message from channel
   *
   * @param out_msg Out message
   *
   * @return S_OK or E_EMPTY
   */
  status_t recv(void*& recvd_msg);

  /**
   * Allocate message (in shared memory) for
   * exchange on channel
   *
   *
   * @return Pointer to message
   */
  void* alloc_msg();

  /**
   * Free message allocated with alloc_msg
   *
   *
   * @return S_OK or E_INVAL
   */
  status_t free_msg(void*);

  /**
   * Used to unblock a thread waiting on a recv
   *
   */
  void unblock_threads();

  /**
   * Shutdown handling
   *
   */
  void set_shutdown() { _shutdown = true; }

  /**
   * Determine if shutdown in progress
   *
   *
   * @return True if so
   */
  bool shutdown() const { return _shutdown; }

 private:
  void initialize_data_structures();

 private:
  bool _shutdown = false;
  bool _master;
  Shared_memory* _shmem_fifo_m2s;
  Shared_memory* _shmem_fifo_s2m;
  Shared_memory* _shmem_slab_ring;
  Shared_memory* _shmem_slab;

  queue_t* _in_queue = nullptr;
  queue_t* _out_queue = nullptr;
  queue_t* _slab_ring = nullptr;
};

}  // namespace UIPC
}  // namespace Core

#endif
