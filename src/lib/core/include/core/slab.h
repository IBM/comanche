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



/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __COMMON_SLAB_H__
#define __COMMON_SLAB_H__

#include <common/chksum.h>
#include <common/memory.h>
#include <core/lazy_region.h>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

namespace Core
{
namespace Slab
{
/**
 * Simple slab allocator using POSIX runtime new/delete
 *
 */
template <typename T>
class CRuntime : public Common::Base_slab_allocator {
 public:
  CRuntime(size_t slots = -1, bool exact = false) {}
  void* alloc() { return (T*) ::malloc(sizeof(T)); }
  size_t free(void* ptr) {
    assert(ptr);
    ::free(ptr);
    return sizeof(T);
  }
  bool is_reconstructed() { return false; }
};

/**
 * In-place header for each element of the slab allocator
 *
 */
struct __BasicElementHeader {
  union {
    struct {
      bool used : 1;
      unsigned resv : 7;
    };
    uint8_t flags;
  };
} __attribute__((packed));

template <typename V>
class __BasicElement { /**< order and packing is important */
 public:
  __BasicElementHeader hdr; /**< common header */
  V val;                    /**< value itself */
} __attribute__((packed));

/**
 * Serializable slab allocator which can be re-initialized from a block
 * of memory (persistent or not)
 *
 */
template <typename T = void*,
          template <typename U> class Element = Slab::__BasicElement>
class Allocator : public Common::Base_slab_allocator {
 private:
  static const bool option_DEBUG = true;    /**< toggle to activate debugging */
  static const unsigned MIN_ELEMENTS = 128; /**< sanity bounds */

  // 64 byte header
  struct Header {
    char magic[8];
    size_t region_size;
    uint64_t slots;
    uint64_t max_slots;
    uint64_t slot_size;
    char label[24];
  } __attribute__((packed));

  void* _region;
  unsigned _page_shift;
  std::vector<Element<T>*> _free_slots; /**< this is not written out; its
                                           hdr.used as in-memory index */
  Element<T>* _slot_array; /**< data that can be implicitly serialized */
  Header* _header;
  bool _reconstructed;

 public:
  /**
   * Constructor.  Attempts to recreate from existing region (using the magic
   * number as a hint) or initializes new allocator.
   *
   * @param region Pointer to region (owner to clean up).
   * @param region_size Size of region in bytes
   * @param label Label to save in header
   * @param as_new Force new instantiation if set to true
   * @param page_shift Shift for IO blocks
   *
   */
  Allocator(void* region, size_t region_size, std::string label, bool as_new,
            unsigned page_shift = 12)
      : _region(region),
        _page_shift(page_shift),
        _slot_array((Element<T>*) (((addr_t) _region) + sizeof(Header))),
        _header((Header*) region)  // header is start of region
  {
    static_assert(sizeof(Header) == 64, "Header should be 64 bytes");
    assert(_header);

    _header->magic[6] = '\0';

    if (option_DEBUG) PLOG("Slab magic=%s", _header->magic);

    // recover existing state
    if ((strncmp(_header->magic, "_SLAB_", 6) == 0) && !as_new) {
      /* already initialized */
      PLOG("Reconstructed slab: region_size=%ld, slots=%ld, max_slots=%ld, "
           "slot_size=%ld, label=%s",
           _header->region_size, _header->slots, _header->max_slots,
           _header->slot_size, _header->label);

      // rebuild free slot list
      for (size_t i = 0; i < _header->slots; i++) {
        Element<T>* e = &_slot_array[i];
        if (!e->hdr.used) _free_slots.push_back(e);
      }
      _reconstructed = true;
    }
    else {
      __builtin_memset(_header, 0, sizeof(Header));
      __builtin_memset((void*) _slot_array, 0,
                       sizeof(Element<T>) * _header->max_slots);

      strncpy(_header->magic, "_SLAB_", 6);
      _header->magic[6] = '\0';
      strncpy(_header->label, label.c_str(), sizeof(_header->label));

      _header->region_size = region_size;
      _header->max_slots = (region_size - sizeof(Header)) / sizeof(Element<T>);
      assert(_header->max_slots > 0);

      if (_header->max_slots < MIN_ELEMENTS) /* sanity check */
        throw new Constructor_exception("too small a region");

      _header->slot_size = sizeof(Element<T>);
      _header->slots = 0;

      PLOG("New slab: region_size=%ld, slots=%ld, max_slots=%ld, "
           "slot_size=%ld, label=%s",
           _header->region_size, _header->slots, _header->max_slots,
           _header->slot_size, _header->label);

      _reconstructed = false;
    }
  }

  /**
   * Destructor
   *
   *
   */
  virtual ~Allocator() {}

  /**
   * Determine how much memory to allocate for N slots
   *
   * @param num_slots Number of slots
   *
   * @return Required memory size in bytes
   */
  static size_t determine_size(size_t num_slots) {
    assert(num_slots > 0);
    return (sizeof(Header) + (sizeof(Element<T>) * (num_slots + 1)));
  }

  /**
   * Determine how much memory is needed without type T
   *
   * @param num_slots
   * @param slot_size
   *
   * @return
   */
  static size_t determine_size(size_t num_slots, size_t slot_size) {
    assert(num_slots > 0);
    return (sizeof(Header) +
            ((sizeof(__BasicElementHeader) + slot_size) * (num_slots + 1)));
  }

  /**
   * Get count of number of slots
   *
   *
   * @return Number of slots of type T
   */
  size_t num_slots() const { return _header->max_slots; }

  /**
   * Get number of used slots
   *
   *
   * @return Number of used slots
   */
  size_t used_slots() const {
    size_t count = 0;
    for (size_t i = 0; i < _header->slots; i++) {
      Element<T>* slot = &_slot_array[i];
      if (slot->hdr.used) {
        count++;
      }
    }
    return count;
  }

  /**
   * Determine if the slab was reconstructed or not
   *
   *
   * @return True if reconstructed; false if new init
   */
  bool is_reconstructed() { return _reconstructed; }

  /**
   * Allocate an element from the slab
   *
   *
   * @return Pointer to newly allocated element. NULL when memory runs out.
   */
  void* alloc() {
    assert(_header);
    assert(_header->max_slots > 0);

    if (_free_slots.size() > 0) {
      Element<T>* slot = _free_slots.back();
      if (option_DEBUG)
        PDBG("picked up free slot (%p, flags=%x)", (void*) slot,
             slot->hdr.flags);
      assert(slot);
      assert(slot->hdr.used == false);
      slot->hdr.used = true;
      _free_slots.pop_back();
      return &slot->val;
    }
    else {
      if (option_DEBUG)
        PLOG("adding new slot (array_len=%ld)...", _header->slots);
      if (_header->slots >= _header->max_slots) {
        PERR("max slots (%ld) exceeded", _header->max_slots);
        throw API_exception("Slab allocator (%s) run out of memory!",
                            _header->label);
      }

      Element<T>* slot = &_slot_array[_header->slots];
      assert(slot);
      memset((void*) slot, 0, sizeof(Element<T>));  // could be removed.
      slot->hdr.used = true;
      _header->slots++;

      return &slot->val;
    }
    assert(0);
  }

  /**
   * Free a previously allocated element
   *
   * @param elem Pointer to allocated element
   */
  size_t free(void* pval) {
    // bounds check pointer
    if ((((addr_t) pval) < ((addr_t) _region)) ||
        (((addr_t) pval) > (((addr_t) _region) + _header->region_size))) {
      PWRN("free on invalid pointer (%p)", pval);
      return -1;
    }

    if (option_DEBUG) PDBG("freeing slab element: %p", pval);

    Element<T>* slot = (Element<T>*) (((addr_t) pval) - sizeof(slot->hdr));
    slot->hdr.used = false;

    _free_slots.push_back(slot);
    return sizeof(Element<T>);
  }

  /**
   * Apply functor to slots (up to last used slot)
   *
   * @param functor void f(pointer to slot, size, used flag)
   */
  void apply(std::function<void(void*, size_t, bool)> functor) {
    for (size_t i = 0; i < _header->slots; i++) {
      Element<T>* slot = &_slot_array[i];
      functor(&slot->val, _header->slot_size, slot->hdr.used);
    }
  }

  /**
   * Return number of free slots in currently allocated memory
   *
   *
   * @return Number of free slots
   */
  size_t free_slots() const { return _free_slots.size(); }

  void* get_first_element() { return &_slot_array[0].val; }

 public:
  /**
   * Get the memory limits for used pages
   *
   * @param bottom
   * @param top
   */
  void get_memory_limits(addr_t& bottom, addr_t& top) {
    bottom = (addr_t) _region;
    top = ((addr_t) &_slot_array[_header->slots]);
  }

  /**
   * Dump the status of the slab
   *
   */
  void dump_info() {
    addr_t base = (addr_t) _region;
    addr_t top = base + _header->region_size;

    PLOG("%s", "---------------------------------------------------");
    PLOG("HEADER: magic         (%s) ", _header->magic);
    PLOG("      : slots         (%ld)", _header->slots);
    PLOG("      : max slots     (%ld)", _header->max_slots);
    PLOG("      : slot size     (%ld)", _header->slot_size);
    PLOG("      : label         (%s)", _header->label);
    PLOG("      : memory range  (%p-%p) %ld KB", (void*) _header, (void*) top,
         REDUCE_KB((top - base)));
    PLOG("      : chksum        (%x)",
         Common::chksum32(_header, _header->region_size));
    PLOG("%s", "---------------------------------------------------");

#ifdef SHOW_ENTRIES
    for (unsigned i = 0; i < _header->slots; i++) {
      if (i == 100) {
        PLOG("...");
        break;  // short circuit
      }

      Element<T>* slot = &_slot_array[i];

      constexpr size_t val_size = sizeof(T);
      std::stringstream sstr;
      byte* v = (byte*) &slot->val;
      for (size_t i = 0; i < val_size; i++) {
        sstr << std::hex << std::setw(2) << std::setfill('0') << int(*v) << " ";
        v++;
        if (i > 10) break;
      }

      PLOG("\t[%u]: %s %p : %s", i, slot->hdr.used ? "USED" : "EMPTY",
           (void*) &slot->val, sstr.str().c_str());
    }
#endif

    PLOG("%s", "---------------------------------------------------");
    PLOG(" Volatile status: _free_slots.size = %ld", _free_slots.size());
    unsigned i = 0;
    for (auto fs : _free_slots) {
      PLOG("FREE slot entry: %p", (void*) fs);
      i++;
      if (i == 100) {
        PLOG("%s", "...");
        break;
      }
    }
    PLOG("%s", "---------------------------------------------------");
  }

  /**
   * For debugging purposes.  Sort the free slot vector.
   *
   */
  void __dbg_sort_free_slots() {
    sort(_free_slots.begin(), _free_slots.end(),
         [](Element<T>* i, Element<T>* j) -> bool {
           return ((addr_t) i < (addr_t) j);
         });
  }

  /**
   * For debugging purposes. Get reference to slot vector
   *
   *
   * @return
   */
  std::vector<Element<T>*>& __dbg_slots() { return _free_slots; }
};

}  // namespace Slab
}  // namespace Core

#endif
