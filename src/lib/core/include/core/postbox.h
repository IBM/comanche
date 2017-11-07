#ifndef __CORE_POSTBOX_H__
#define __CORE_POSTBOX_H__

#include <common/utils.h>

namespace Core {
  
template <typename T = void*>
class Mpmc_postbox
{
  static_assert(sizeof(T) == sizeof(void*),"T must be 64-bit");
#ifdef __CUDACC__
  static_assert(sizeof(unsigned long long int) == sizeof(T), "T must be 64-bit");
#endif
  
public:
  Mpmc_postbox(void * shared_buffer, size_t shared_buffer_size, bool master = false) {
    assert(shared_buffer);
    assert(shared_buffer_size % sizeof(T) == 0);
    assert(check_aligned(shared_buffer_size, sizeof(T)));

    _slots = static_cast<T*>(shared_buffer);
    _num_slots = shared_buffer_size / sizeof(T);
    assert(_num_slots > 0);

    if(master) {
      memset(_slots, 0, shared_buffer_size);
    }
    PLOG("num slots = %u", _num_slots);
  }  
  
  static size_t memory_size(unsigned num_slots) {
    return sizeof(T) * num_slots;
  }
  
  bool post(T val)
  {
    unsigned next = _next_free;
    unsigned attempts = 0;

    do {
      if(_slots[next]) {
        next++;
        if(next == _num_slots) next = 0;
      }
#ifdef __CUDACC__
      else if(atomicCAS(&_slots[next], 0, val) == 0)
#else
      else if(__sync_bool_compare_and_swap(&_slots[next], 0, val))
#endif
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
#ifdef __CUDACC__        
        if(atomicCAS(&_slots[next], val, 0) == val)
#else
        if(__sync_bool_compare_and_swap(&_slots[next], val, 0))
#endif
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
  
private:
  T*       _slots;
  unsigned _num_slots = 0;
  unsigned _next_free = 0;
  unsigned _next_to_collect = 0;
  
} __attribute__((packed));

} // namespace Core

#endif
