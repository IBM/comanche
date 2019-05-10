/*
   Copyright [2017-2019] [IBM Corporation]
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
#ifndef __CLIENT_FABRIC_TRANSPORT_H__
#define __CLIENT_FABRIC_TRANSPORT_H__

#include <api/fabric_itf.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include "dawn_client_config.h"

namespace Dawn
{
namespace Client
{
class Fabric_transport {
  friend class Dawn_client;

  const bool option_DEBUG = Dawn::Global::debug_level > 1;

 public:
  using Transport       = Component::IFabric_client;
  using buffer_t        = Buffer_manager<Transport>::buffer_t;
  using memory_region_t = Component::IFabric_memory_region *;

  Fabric_transport(Component::IFabric_client *fabric_connection)
      : _transport(fabric_connection), _bm(fabric_connection)
  {
    _max_inject_size = _transport->max_inject_size();
  }

  ~Fabric_transport() {}

  static Component::IFabric_op_completer::cb_acceptance completion_callback(
      void *        context,
      status_t      st,
      std::uint64_t completion_flags,
      std::size_t   len,
      void *        error_data,
      void *        param)
  {
    if (unlikely(st != S_OK))
      throw Program_exception(
          "poll_completions failed unexpectedly (st=%d) (cf=%lx)", st,
          completion_flags);

    if (*((void **) param) == context) {
      *((void **) param) = nullptr;
      return Component::IFabric_op_completer::cb_acceptance::ACCEPT;
    }
    else {
      return Component::IFabric_op_completer::cb_acceptance::DEFER;
    }
  }

  /**
   * Wait for completion of a IO buffer posting
   *
   * @param iob IO buffer to wait for completion of
   */
  void wait_for_completion(void *wr)
  {
    void *p = wr;
    while (p) {
      _transport->poll_completions_tentative(completion_callback, &p);
    }
  }

  /**
   * Forwarders that allow us to avoid exposing _transport and _bm
   *
   */
  inline memory_region_t register_memory(void *base, size_t len)
  {
    return _transport->register_memory(base, len, 0, 0);
  }

  inline void *get_memory_descriptor(memory_region_t region)
  {
    return _transport->get_memory_descriptor(region);
  }

  inline void deregister_memory(memory_region_t region)
  {
    _transport->deregister_memory(region);
  }

  inline void post_send(const ::iovec *first,
                        const ::iovec *last,
                        void **        descriptors,
                        void *         context)
  {
    _transport->post_send(first, last, descriptors, context);
  }

  inline void post_recv(const ::iovec *first,
                        const ::iovec *last,
                        void **        descriptors,
                        void *         context)
  {
    _transport->post_recv(first, last, descriptors, context);
  }

  inline size_t max_message_size() const {
    return _transport->max_message_size();
  }

  /**
   * Post send (one or two buffers) and wait for completion.
   *
   * @param iob First IO buffer
   * @param iob_extra Second IO buffer
   */
  void sync_send(buffer_t *iob, buffer_t *iob_extra = nullptr)
  {
    if (iob_extra) {
      iovec v[2]   = {*iob->iov, *iob_extra->iov};
      void *desc[] = {iob->desc, iob_extra->desc};

      post_send(&v[0], &v[2], desc, iob);
    }
    else {
      post_send(iob->iov, iob->iov + 1, &iob->desc, iob);
    }

    wait_for_completion(iob);
    iob->reset_length();
  }

  /**
   * Perform inject send (fast for small packets)
   *
   * @param iob Buffer to send
   */
  void sync_inject_send(buffer_t *iob)
  {
    auto len = iob->length();
    if (len <= _max_inject_size) {
      /* when this returns, iob is ready for immediate reuse */
      _transport->inject_send(iob->base(), iob->length());
    }
    else {
      /* too big for inject, do plain send */
      post_send(iob->iov, iob->iov + 1, &iob->desc, iob);
      wait_for_completion(iob);
    }

    iob->reset_length();
  }

  /**
   * Post send (one or two buffers) and wait for completion.
   *
   * @param iob First IO buffer
   * @param iob_extra Second IO buffer
   */
  void post_send(buffer_t *iob, buffer_t *iob_extra = nullptr)
  {
    if (iob_extra) {
      iovec v[2]   = {*iob->iov, *iob_extra->iov};
      void *desc[] = {iob->desc, iob_extra->desc};

      post_send(&v[0], &v[2], desc, iob);
    }
    else {
      post_send(iob->iov, iob->iov + 1, &iob->desc, iob);
    }
  }

  /**
   * Post receive then wait for completion before returning.
   * Use after post_send may lead to poor performance if the response
   * arrives before the receive buffer is posted.
   *
   * @param iob IO buffer
   */
  void sync_recv(buffer_t *iob)
  {
    sync_recv_0(iob);
    sync_recv_1(iob);
  }

  /**
   * Post receive (and do not wait for completion before returning).
   *
   * @param iob IO buffer
   */
  void sync_recv_0(buffer_t *iob)
  {
    if (option_DEBUG)
      PLOG("sync_recv: (%p, %p, base=%p, len=%lu)", iob, iob->desc,
           iob->iov->iov_base, iob->iov->iov_len);

    iob->reset_length();
    post_recv(iob->iov, iob->iov + 1, &iob->desc, iob);
  }

  /**
   * Wait for (receive) completion before returning.
   *
   * @param iob IO buffer
   */
  void sync_recv_1(buffer_t *iob)
  {
    wait_for_completion(iob);
  }

  void post_recv(buffer_t *iob)
  {
    if (option_DEBUG)
      PLOG("post_recv: (%p, %p, base=%p, len=%lu)", iob, iob->desc,
           iob->iov->iov_base, iob->iov->iov_len);

    iob->reset_length();
    post_recv(iob->iov, iob->iov + 1, &iob->desc, iob);
  }

  Component::IKVStore::memory_handle_t register_direct_memory(void * region,
                                                              size_t region_len)
  {
    // if (!check_aligned(region, 64))
    //   throw API_exception("register_direct_memory: region should be aligned");

    auto mr        = register_memory(region, region_len);
    auto desc      = get_memory_descriptor(mr);
    auto buffer    = new buffer_t(region_len);
    buffer->iov    = new iovec{(void *) region, region_len};
    buffer->region = mr;
    buffer->desc   = desc;

    if (option_DEBUG)
      PLOG("register_direct_memory (%p, %lu, mr=%p, desc=%p)", region,
           region_len, mr, desc);

    return reinterpret_cast<Component::IKVStore::memory_handle_t>(buffer);
  }

  status_t unregister_direct_memory(Component::IKVStore::memory_handle_t handle)
  {
    buffer_t *buffer = reinterpret_cast<buffer_t *>(handle);
    assert(buffer->check_magic());

    _transport->deregister_memory(buffer->region);
    return S_OK;
  }

  inline auto allocate() { return _bm.allocate(); }
  inline void free_buffer(buffer_t *buffer) { _bm.free(buffer); }

 protected:
  Transport *               _transport;
  size_t                    _max_inject_size;
  Buffer_manager<Transport> _bm; /*< IO buffer manager */
};

}  // namespace Client
}  // namespace Dawn

#endif  //__CLIENT_FABRIC_TRANSPORT_H__
