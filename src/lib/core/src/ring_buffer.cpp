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

#if defined(__x86_64__)

#include "ring_buffer.h"
#include <common/exceptions.h>
#include <rte_errno.h>
#include <rte_ring.h>
#include <string.h>

namespace Core
{
Ring_buffer_base::Ring_buffer_base(std::string name, size_t size) {
  _rte_ring = rte_ring_create(name.c_str(), /* debug name */
                              size,         /* q depth */
                              SOCKET_ID_ANY, 0);

  if (_rte_ring == nullptr && rte_errno == 17)
    throw General_exception("rte_ring_create failed (name=%s) exists!",
                            name.c_str());
  else if (_rte_ring == nullptr)
    throw General_exception(
        "rte_ring_create failed (size=%lu) for Ring_buffer_base class (%d)",
        size, rte_errno);

  if (option_DEBUG)
    PLOG("Ring_buffer_base [%s]: allocated at %p size=%lu", name.c_str(),
         _rte_ring, size);
}

Ring_buffer_base::Ring_buffer_base(std::string name, int index, size_t size) {
  char name_sz[255];
  sprintf(name_sz, "%s-%d", name.c_str(), index);

  _rte_ring = rte_ring_create(name_sz, /* debug name */
                              size,    /* q depth */
                              SOCKET_ID_ANY, 0);
  if (_rte_ring == nullptr && rte_errno == 17)
    throw General_exception("rte_ring_create failed (name=%s) exists!",
                            name_sz);
  else if (_rte_ring == nullptr)
    throw General_exception(
        "rte_ring_create failed (size=%lu) for Ring_buffer_base class (%d)",
        size, rte_errno);

  if (option_DEBUG)
    PLOG("Ring_buffer_base [%s]: allocated at %p size=%lu", name_sz, _rte_ring,
         size);
}

Ring_buffer_base::~Ring_buffer_base() {
  if (option_DEBUG) PLOG("Ring_buffer_base: freeing at %p", _rte_ring);
  rte_ring_free(_rte_ring);
}

int Ring_buffer_base::full() {
  assert(_rte_ring);
  return rte_ring_full(_rte_ring);
}

int Ring_buffer_base::empty() {
  assert(_rte_ring);
  return rte_ring_empty(_rte_ring);
}

int Ring_buffer_base::sp_enqueue(void* elem) {
  assert(_rte_ring);
  assert(elem);
  return rte_ring_sp_enqueue(_rte_ring, (void*) elem);
}

int Ring_buffer_base::sc_dequeue(void*& elem) {
  assert(_rte_ring);
  return rte_ring_sc_dequeue(_rte_ring, (void**) &elem);
}

int Ring_buffer_base::mp_enqueue(void* elem) {
  assert(_rte_ring);
  assert(elem);
  return rte_ring_mp_enqueue(_rte_ring, (void*) elem);
}

int Ring_buffer_base::mc_dequeue(void*& elem) {
  assert(_rte_ring);
  return rte_ring_mc_dequeue(_rte_ring, (void**) &elem);
}

std::atomic<uint64_t> Ring_buffer_base::_static_name_index;

}  // namespace Core

#endif
