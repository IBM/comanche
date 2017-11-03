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

#include <sys/stat.h>
#include <string>
#include <mutex>

/* Expose as C because we want other programming languages to
   interface onto this API */

extern "C"
{
  typedef void * channel_t;

  /** 
   * Channel is bi-directional, user-level, lock-free exchange of
   * fixed sized messages.  Receiving is polling-based. It does not
   * define the message protocol which can be Protobuf etc.  Channel
   * is a lock-free FIFO in shared memory for passing pointers
   * together with a slab allocator (also lock-free and thread-safe
   * across both sides) for fixed sized messages.  Both sides map to
   * the same virtual address; this is negogiated.
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
  channel_t uipc_create_channel(const char * path_name,
                                size_t message_size,
                                size_t queue_size);

  /** 
   * Connect to a channel
   * 
   * @param path_name Name of channel (e.g., /tmp/myChannel)
   * @param 
   * 
   * @return Handle to channel or NULL on failure
   */
  channel_t uipc_connect_channel(const char * path_name);

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
  void * uipc_alloc_message(channel_t channel);

  /** 
   * Free message
   * 
   * @param channel Associated channel
   * @param message Message
   * 
   * @return 
   */
  status_t uipc_free_message(channel_t channel, void * message);


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
  status_t uipc_pop(channel_t channel, void*& data_out);
}



namespace Core {
namespace UIPC {

class Shared_memory
{
private:
  static constexpr bool option_DEBUG = true;
  
public:
  Shared_memory(std::string name, size_t n_pages); /*< initiator constructor */
  Shared_memory(std::string name); /*< target constructor */
  
  virtual ~Shared_memory();

  void * get_addr();

  size_t get_size_in_pages() const { return _size_in_pages; }
  size_t get_size() const { return _size_in_pages * PAGE_SIZE; }

private:
  static void * negotiate_addr_create(std::string path_name,
                                      size_t size_in_bytes);
  
  static void * negotiate_addr_connect(std::string path_name,
                                       size_t * size_in_bytes_out);


  void open_shared_memory(std::string path_name);
  
private:
  std::string _name;
  void *      _vaddr;
  size_t      _size_in_pages;
};


class Channel
{
private:
  static constexpr bool option_DEBUG = true;

public:
  Channel(std::string name, size_t message_size, size_t queue_size);
  Channel(std::string name);
  virtual ~Channel();

private:
  Shared_memory * _shmem_fifo;
  Shared_memory * _shmem_slab_ring;
  Shared_memory * _shmem_slab;
  
};

}} // Core::UIPC

#endif
