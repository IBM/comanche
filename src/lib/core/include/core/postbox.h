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

#ifndef __CORE_POSTBOX_H__
#define __CORE_POSTBOX_H__

#include <common/utils.h>

#ifdef __CUDACC__
#define CUDA_DEVICE_MEMBER __device__
#else
#define CUDA_DEVICE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_DUAL_MEMBER __host__ __device__
#else
#define CUDA_DUAL_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_SHARED __shared__
#else
#define CUDA_SHARED
#endif

namespace Core {

/** 
 * Thread-safe post box for MP-MC exchange of 64bit values (e.g.,
 * pointers). Initial tests show around 18M exchanges per second.
 * 
 */
template <typename T = void*>
class Mpmc_postbox
{
  static_assert(sizeof(T) == sizeof(void*),"T must be 64-bit");
#ifdef __CUDACC__
  static_assert(sizeof(unsigned long long int) == sizeof(T), "T must be 64-bit");
#endif
  
public:

  /** 
   * Constructor
   * 
   * @param shared_buffer Pointer to shared buffer area
   * @param shared_buffer_size Size of shared buffer in bytes
   * @param master Set if master role
   */
  CUDA_DUAL_MEMBER Mpmc_postbox(void * shared_buffer, size_t shared_buffer_size, bool master = false) {
    assert(shared_buffer);
    assert(shared_buffer_size % sizeof(T) == 0);

    _slots = static_cast<T*>(shared_buffer);
    _num_slots = shared_buffer_size / sizeof(T);
    assert(_num_slots > 0);

    if(master) {
      memset((void*)_slots, 0, shared_buffer_size);
    }
  }  

  /** 
   * Statically determine memory size for a specific number of slots
   * 
   * @param num_slots Number of slots of type T (sizeof(T)==sizeof(void*))
   * 
   * @return Size in bytes
   */
  CUDA_DUAL_MEMBER static size_t memory_size(unsigned num_slots) {
    return sizeof(T) * num_slots;
  }

  /** 
   * Post a value
   * 
   * @param val Value to post
   * 
   * @return True on success, false on full.
   */
  bool post(T val)
  {
    unsigned next = _next_free;
    unsigned attempts = 0;

    do {
      if(_slots[next]) {
        next++;
        if(next == _num_slots) next = 0;
      }
      else if(__sync_bool_compare_and_swap(&_slots[next], 0, val))
      {
        _next_free = next ++;
        if(_next_free == _num_slots) _next_free = 0;
        return true;
      }
      attempts++;
    }
    while(attempts < (_num_slots * 2));
    return false;
  }

  bool collect(T& out_val)
  {
    unsigned next = _next_to_collect;
    unsigned attempts = 0;
    do {
      if(_slots[next] == 0) {
        next++;
        if(next == _num_slots) next = 0;
      }
      else {
        T val = _slots[next];
        PLOG("found something %lu", val);

        // int64_t goo = __sync_fetch_and_add(&_slots[next], 1);
        // PLOG("goo = %ld", goo);
        if(__sync_bool_compare_and_swap((unsigned long long *) &_slots[next], (unsigned long long) val, 0))  {
          _slots[next] = 0;
          out_val = val;
          _next_to_collect = next + 1;
          if(_next_to_collect == _num_slots) _next_to_collect = 0;
          return true;
        }
        else {
          PLOG("but CAS failed");
          sleep(1);
        }
      }
      attempts++;
    } while(attempts < (_num_slots * 2));
    return false;
  }

#ifdef __CUDACC__
  CUDA_DEVICE_MEMBER bool post_device(T val)
  {
    unsigned next = _next_free;
    unsigned attempts = 0;

    do {
      if(_slots[next]) {
        next++;
        if(next == _num_slots) next = 0;
      }
      else if(atomicCAS((unsigned long long int *) &_slots[next], 0ULL, (unsigned long long int) val) == 0)
      {
        printf("GPU-side posted:%lu\n", _slots[next]);
        _next_free = next ++;
        if(_next_free == _num_slots) _next_free = 0;
        return true;
      }
      attempts++;
    }
    while(attempts < (_num_slots * 2));
    printf("GPU-side post failed\n");
    return false;
  }
  
  /** 
   * Collect a message
   * 
   * @param out_val Out value
   * 
   * @return True on success, false on empty
   */
  CUDA_DEVICE_MEMBER bool collect_device(T& out_val)
  {
    unsigned next = _next_to_collect;
    unsigned attempts = 0;
    do {
      if(_slots[next] == 0) {
        next++;
        if(next == _num_slots) next = 0;
      }
      else {
        T val = _slots[next];
        if(atomicCAS((unsigned long long int*)&_slots[next], (unsigned long long int)val, 0ULL) == val)
          {
            out_val = val;
            _next_to_collect = next + 1;
            if(_next_to_collect == _num_slots) _next_to_collect = 0;
            return true;
          }        
      }
      attempts++;
    } while(attempts < (_num_slots * 2));
    return false;
  }
#endif
  

  
private:
  volatile T* _slots;         /* points to shared memory */
  unsigned    _num_slots = 0; /* these members will be in separate memory areas */
  unsigned    _next_free = 0;
  unsigned    _next_to_collect = 0;
  
} __attribute__((packed));

} // namespace Core

#endif
