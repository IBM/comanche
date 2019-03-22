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


#ifndef __NUPM_DAX_DATA_H__
#define __NUPM_DAX_DATA_H__

#include <common/types.h>
#include <common/utils.h>
#include <libpmem.h>
#include "pm_lowlevel.h"

#define DM_REGION_MAGIC 0xC0070000
#define DM_REGION_NAME_MAX_LEN 1024
#define DM_REGION_VERSION 2

namespace nupm
{
struct DM_undo_log {
  static constexpr unsigned MAX_LOG_COUNT = 4;
  static constexpr unsigned MAX_LOG_SIZE  = 64;
  struct log_entry_t {
    byte   log[MAX_LOG_SIZE];
    void * ptr;
    size_t length; /* zero indicates log freed */
  };

 public:
  void log(void *ptr, size_t length)
  {
    assert(length > 0);
    assert(ptr);

    if (length > MAX_LOG_SIZE)
      throw API_exception("log length exceeds max. space");

    for (unsigned i = 0; i < MAX_LOG_COUNT; i++) {
      if (_log[i].length == 0) {
        _log[i].length = length;
        _log[i].ptr    = ptr;
        pmem_memcpy_nodrain(_log[i].log, ptr, length);
        pmem_flush(&_log[i], sizeof(log_entry_t));
        return;
      }
    }
    throw API_exception("undo log full");
  }

  void clear_log()
  {
    for (unsigned i = 0; i < MAX_LOG_COUNT; i++) _log[i].length = 0;
  }

  void check_and_undo()
  {
    for (unsigned i = 0; i < MAX_LOG_COUNT; i++) {
      if (_log[i].length > 0) {
        PLOG("undo log being applied (ptr=%p, len=%lu).", _log[i].ptr,
             _log[i].length);
        pmem_memcpy_persist(_log[i].ptr, _log[i].log, _log[i].length);
        _log[i].length = 0;
      }
    }
  }

 private:
  log_entry_t _log[MAX_LOG_COUNT];
} __attribute__((packed));

struct DM_region {
 public:
  uint32_t offset_GB;
  uint32_t length_GB;
  uint64_t region_id;

 public:
  /* re-zeroing constructor */
  DM_region() : length_GB(0), region_id(0) { assert(check_aligned(this, 8)); }

  void initialize(size_t space_size)
  {
    offset_GB = 0;
    length_GB = space_size / GB(1);
    region_id = 0; /* zero indicates free */
  }
} __attribute__((packed));

class DM_region_header {
 private:
  static constexpr uint16_t DEFAULT_MAX_REGIONS = 1024;

  uint32_t    _magic;         // 4
  uint32_t    _version;       // 8
  uint64_t    _device_size;   // 16
  uint32_t    _region_count;  // 20
  uint32_t    _resvd;         // 24
  uint8_t     _padding[40];   // 64
  DM_undo_log _undo_log;
  DM_region   _regions[];

 public:
  /* Rebuilding constructor */
  DM_region_header(size_t device_size)
  {
    reset_header(device_size);

    _region_count       = DEFAULT_MAX_REGIONS;
    DM_region *region_p = region_table_base();
    /* initialize first region with all capacity */
    region_p->initialize(device_size - GB(1));
    _undo_log.clear_log();
    region_p++;

    for (uint16_t r = 1; r < _region_count; r++) {
      new (region_p) DM_region();
      _undo_log.clear_log();
      region_p++;
    }
    major_flush();
  }

  void check_undo_logs()
  {
    PLOG("Checking undo logs..");
    _undo_log.check_and_undo();
  }

  void debug_dump()
  {
    PINF("DM_region_header:");
    PINF(
        " magic [0x%8x]\n version [%u]\n device_size [%lu]\n region_count [%u]",
        _magic, _version, _device_size, _region_count);
    PINF(" base [%p]", this);

    for (uint16_t r = 0; r < _region_count; r++) {
      auto reg = _regions[r];
      if (reg.region_id > 0) {
        PINF(" - USED: %lu (%lx-%lx)", reg.region_id,
             GB_to_bytes(reg.offset_GB),
             GB_to_bytes(reg.offset_GB + reg.length_GB) - 1);
        assert(reg.length_GB > 0);
      }
      else if (reg.length_GB > 0) {
        PINF(" - FREE: %lu (%lx-%lx)", reg.region_id,
             GB_to_bytes(reg.offset_GB),
             GB_to_bytes(reg.offset_GB + reg.length_GB) - 1);
      }
    }
  }

  void *get_region(uint64_t region_id, size_t *out_size)
  {
    if (region_id == 0) throw API_exception("invalid region_id");

    for (uint16_t r = 0; r < _region_count; r++) {
      auto reg = _regions[r];
      if (reg.region_id == region_id) {
        PLOG("found matching region (%lx)", region_id);
        if (out_size) *out_size = (((uintptr_t) reg.length_GB) << 30);
        return (((uintptr_t) reg.offset_GB) << 30) + arena_base();
      }
    }
    return nullptr; /* not found */
  }

  void erase_region(uint64_t region_id)
  {
    if (region_id == 0) throw API_exception("invalid region_id");

    for (uint16_t r = 0; r < _region_count; r++) {
      DM_region *reg = &_regions[r];
      if (reg->region_id == region_id) {
        reg->region_id = 0; /* power-fail atomic */
        mem_flush(&reg->region_id, sizeof(reg->region_id));
        return;
      }
    }
    throw API_exception("region (%lu) not found", region_id);
  }

  void *allocate_region(uint64_t region_id, unsigned size_in_GB)
  {
    if (region_id == 0) throw API_exception("invalid region_id");

    for (uint16_t r = 0; r < _region_count; r++) {
      auto reg = _regions[r];
      if (reg.region_id == region_id)
        throw API_exception("region_id already exists");
    }
    // TODO make crash-consistent
    uint32_t new_offset;
    bool     found = false;
    for (uint16_t r = 0; r < _region_count; r++) {
      DM_region *reg = &_regions[r];
      if (reg->region_id == 0 && reg->length_GB >= size_in_GB) {
        if (reg->length_GB == size_in_GB) {
          /* exact match */
          void *rp =
              (void *) ((((uintptr_t) reg->offset_GB) << 30) + arena_base());
          // zero region
          // pmem_memset_persist(rp, 0, GB(((uintptr_t)size_in_GB)));
          tx_atomic_write(reg, region_id);
          return rp;
        }
        else {
          /* cut out */
          new_offset = reg->offset_GB;

          uint16_t changed_length = reg->length_GB - size_in_GB;
          uint16_t changed_offset = reg->offset_GB + size_in_GB;

          for (uint16_t r = 0; r < _region_count; r++) {
            DM_region *reg_n = &_regions[r];
            if (reg_n->region_id == 0 && reg_n->length_GB == 0) {
              void *rp =
                  (void *) ((((uintptr_t) new_offset) << 30) + arena_base());
              //pmem_memset_persist(rp, 0, GB(((uintptr_t) size_in_GB)));
              tx_atomic_write(reg_n, changed_offset, changed_length, reg,
                              new_offset, size_in_GB, region_id);
              return rp;
            }
          }
        }
      }
    }
    if (!found)
      throw General_exception("no more regions (size in GB=%u)", size_in_GB);

    throw General_exception("no spare slots");
  }

  size_t get_max_available() const
  {
    size_t max_size = 0;
    for (uint16_t r = 0; r < _region_count; r++) {
      const DM_region *reg = &_regions[r];
      if (reg->length_GB > max_size) max_size = reg->length_GB;
    }
    return GB_to_bytes(max_size);
  }

  inline size_t GB_to_bytes(unsigned GB) const { return ((size_t) GB) << 30; }

  inline void major_flush()
  {
    nupm::mem_flush(
        this, sizeof(DM_region_header) + (sizeof(DM_region) * _region_count));
  }

  bool check_magic() const
  {
    return (_magic == DM_REGION_MAGIC) && (_version == DM_REGION_VERSION);
  }

 private:
  void tx_atomic_write(DM_region *dst, uint64_t region_id)
  {
    _undo_log.log(&dst->region_id, sizeof(region_id));
    dst->region_id = region_id;
    mem_flush(&dst->region_id, sizeof(region_id));
    _undo_log.clear_log();
  }

  void tx_atomic_write(DM_region *dst0,
                       uint32_t   offset0,
                       uint32_t   size0,
                       DM_region *dst1,
                       uint32_t   offset1,
                       uint32_t   size1,
                       uint64_t   region_id1)
  {
    _undo_log.log(dst0, sizeof(DM_region));
    _undo_log.log(dst1, sizeof(DM_region));

    dst0->offset_GB = offset0;
    dst0->length_GB = size0;
    mem_flush(dst0, sizeof(DM_region));

    dst1->region_id = region_id1;
    dst1->offset_GB = offset1;
    dst1->length_GB = size1;
    mem_flush(dst1, sizeof(DM_region));

    _undo_log.clear_log();
  }

  inline unsigned char *arena_base()
  {
    return (((unsigned char *) this) + GB(1));
  }

  inline DM_region *region_table_base() { return _regions; }

  inline DM_region *region(size_t idx)
  {
    if (idx >= _region_count) return nullptr;
    DM_region *p = (DM_region *) region_table_base();
    return &p[idx];
  }

  void reset_header(size_t device_size)
  {
    _magic       = DM_REGION_MAGIC;
    _version     = DM_REGION_VERSION;
    _device_size = device_size;
    nupm::mem_flush(this, sizeof(DM_region_header));
  }
} __attribute__((packed));

}  // namespace nupm

#endif  //__NUPM_DAX_DATA_H__
