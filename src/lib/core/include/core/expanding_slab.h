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

#ifndef __COMMON_EXPANDING_SLAB_H__
#define __COMMON_EXPANDING_SLAB_H__

#include <algorithm>
#include <vector>

#include <common/chksum.h>
#include <common/memory.h>
#include <core/lazy_region.h>
#include <core/slab.h>

namespace Core
{
namespace Slab
{
/**
 * Slab allocator that can be easily serialized due to compact data
 * arrangement.  It is also capable of dynamically expanding.  This class is
 * used as the slab allocator for tree nodes.  Provides base interface
 * (Common::Base_slab_allocator).
 *
 * NOTE: this allocator is NOT thread-safe.
 */
template <typename T,
          template <typename T> class Element = Slab::__BasicElement>
class Expanding_allocator : public Common::Base_slab_allocator {
 public:
  static constexpr bool DEBUG = false; /**< toggle to activate debugging */

  // 64 byte header
  struct Header {
    char magic[8];
    uint64_t slots;
    uint64_t max_slots;
    uint64_t slot_size;
    char label[32];
  };

 public:
  /**
   * Constructor.
   *
   * @param max_slots Desired max # of slots.  May be more due to page size
   * round up.
   */
  Expanding_allocator(size_t max_slots, std::string label, bool exact = false)
      : _region(
            round_up_page((sizeof(Element<T>) * max_slots) + sizeof(Header))),
        _max_slots(exact ? max_slots
                         : (round_up_page(sizeof(Element<T>) * max_slots)) /
                               sizeof(Element<T>)),
        _slot_array((Element<T>*) (_region.base() + sizeof(Header))) {
    static_assert(sizeof(Header) == 64, "Header should be 64 bytes");
    PDBG("max_slots requested=%ld, actual=%ld", max_slots, _max_slots);
    assert(_max_slots > 0);
    _header = (Header*) _region.base();
    __builtin_memset(_header, 0, sizeof(Header));
    strcpy(_header->magic, "_EXPAND");
    strncpy(_header->label, label.c_str(), sizeof(_header->label));
    _header->max_slots = max_slots;
    _header->slot_size = sizeof(Element<T>);
    _header->slots = 0;
    _callback = NULL;
  }

  /**
   * Destructor
   *
   */
  virtual ~Expanding_allocator() {}

  /**
   * Allocate an element from the slab
   *
   *
   * @return Pointer to newly allocated element. NULL when memory runs out.
   */
  void* alloc() {
    assert(_max_slots > 0);

    if (_free_slots.size() > 0) {
      Element<T>* slot = _free_slots.back();
      if (DEBUG)
        PDBG("picked up free slot (%p, flags=%x)", (void*) slot,
             slot->hdr.flags);
      assert(slot);
      assert(slot->hdr.used == false);
      slot->hdr.used = true;
      _free_slots.pop_back();
      dirty_slot(slot);
      return &slot->val;
    }
    else {
      if (DEBUG) PLOG("adding new slot (array_len=%ld)...", _header->slots);
      if (_header->slots >= _max_slots) {
        PERR("max slots (%ld) exceeded", _max_slots);
        throw API_exception("Expanding_allocator run out of memory!");
      }

      Element<T>* slot = &_slot_array[_header->slots];
      assert(slot);
      __builtin_memset(slot, 0, sizeof(Element<T>));
      slot->hdr.used = true;
      _header->slots++;
      dirty_slot(slot);
      dirty_header();

      if (DEBUG)
        PLOG("Expanding allocator: range(%lx,%lx)", _region.base(),
             _region.base() + _region.mapped_size());

      return &slot->val;
    }
    assert(0);
  }

  /**
   * Free a previously allocated element
   *
   * @param elem Pointer to allocated element
   */
  int free(void* pval) {
    if (DEBUG) PDBG("freeing: %p", pval);
    Element<T>* slot = (Element<T>*) (((addr_t) pval) - sizeof(slot->hdr));
    slot->hdr.used = false;
    _free_slots.push_back(slot);
    dirty_slot(slot);
    return 0;
  }

  /**
   * Return number of free slots in currently allocated memory
   *
   *
   * @return Number of free slots
   */
  size_t free_slots() const { return _free_slots.size(); }

  /**
   * Register a callback for signalling page writes
   *
   * @param callback Callback function
   */
  void register_dirty_callback(std::function<void(addr_t, size_t)> callback) {
    _callback = callback;
  }

 private:
  // TODO - configurable block size

  void dirty_slot(Element<T>* slot) {
    if (_callback) {
      /* work out #pages spanned */
      unsigned spanned = 0;
      addr_t start_page = round_down_page((addr_t) slot);
      addr_t slot_start = ((addr_t) slot);
      addr_t slot_end = ((addr_t) slot) + sizeof(Element<T>);
      while ((start_page << 12) < slot_end) {
        start_page++;
        spanned++;
        assert(spanned < 1000);  // sanity?
      }
      _callback(start_page, spanned);
    }
  }

  void dirty_header() {
    if (_callback) {
      _callback(round_down_page((addr_t) _header), 1);
    }
  }

 public:
  /**
   * Dump the status of the slab
   *
   */
  void __dbg_dump_status() {
    addr_t base = _region.base();
    addr_t top = base + _region.mapped_size();

    PLOG("---------------------------------------------------");
    PLOG("HEADER: magic         (%s) ", _header->magic);
    PLOG("      : slots         (%ld)", _header->slots);
    PLOG("      : max slots     (%ld)", _header->max_slots);
    PLOG("      : slot size     (%ld)", _header->slot_size);
    PLOG("      : label         (%s)", _header->label);
    PLOG("      : memory range  (%p-%p) %ld KB", (void*) _header, (void*) top,
         REDUCE_KB(top - base));
    PLOG("      : mapped pages  (%ld)", _region.mapped_pages());
    PLOG("      : chksum        (%lx)",
         Common::chksum32(_header, Element<T> * _header->max_slots));
    PLOG("---------------------------------------------------");

    for (unsigned i = 0; i < _header->slots; i++) {
      if (i == 100) {
        PLOG("...");
        break;  // short circuit
      }
      Element<T>* slot = &_slot_array[i];
      PLOG("\t[%u]: %s", i, slot->hdr.used ? "USED" : "EMPTY");
    }
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

 private:
  Core::Slab::Lazily_extending_region
      _region; /**< manages the base contiguous region of memory */

  std::vector<Element<T>*> _free_slots; /**< this is not written out; its
                                           hdr.used as in-memory index */
  const size_t _max_slots;
  Element<T>* const _slot_array; /**< data that can be implicitly serialized */
  Header* _header;

  std::function<void(addr_t, size_t)> _callback;
};

}  // namespace Slab
}  // namespace Core

#endif
