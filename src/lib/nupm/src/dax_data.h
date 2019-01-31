#ifndef __NUPM_DAX_DATA_H__
#define __NUPM_DAX_DATA_H__

#include <common/utils.h>
#include "pm_lowlevel.h"

#define DM_REGION_MAGIC 0xC0070000
#define DM_REGION_NAME_MAX_LEN 1024
#define DM_REGION_VERSION 2

namespace nupm
{
  

struct DM_region
{
public:
  uint32_t offset_GB;
  uint32_t length_GB;
  uint64_t region_id;
  
public:
  /* re-zeroing constructor */
  DM_region() : length_GB(0), region_id(0) {
    assert(check_aligned(this, 8));
  }
  
  void initialize(size_t space_size) {
    offset_GB =  0;
    length_GB = space_size / GB(1);
    region_id = 0; /* zero indicates free */
  }
} __attribute__((packed));


class DM_region_header
{  
private:
  static constexpr uint16_t DEFAULT_MAX_REGIONS = 1024;

  struct Undo_t {
    uint64_t valid;
    DM_region * p_region;
    DM_region region;
  } __attribute__((packed));
  
  uint32_t  _magic;
  uint32_t  _version;
  uint64_t  _device_size;
  uint32_t  _region_count;
  uint32_t  _resvd;
  DM_region _regions[];
  
public:
  /* Rebuilding constructor */
  DM_region_header(size_t device_size) {

    reset_header(device_size);

    _region_count = DEFAULT_MAX_REGIONS;
    DM_region * region_p = region_table_base();

    region_p->initialize(device_size - GB(1));
    region_p ++;
    for(uint16_t r=1;r < _region_count; r++) {
      new (region_p) DM_region();
      region_p ++;
    }
    major_flush();
  }

  DM_region_header() {
  }

  void debug_dump() {
    PINF("DM_region_header:");
    PINF(" magic [0x%8x]\n version [%u]\n device_size [%lu]\n region_count [%u]",
         _magic, _version, _device_size, _region_count);
    PINF(" base [%p]", this);

    for(uint16_t r=0;r<_region_count;r++) {
      auto reg = _regions[r];
      if(reg.region_id > 0) {
        PINF(" - USED: %lu (%lx-%lx)",
             reg.region_id,
             GB_to_bytes(reg.offset_GB),
             GB_to_bytes(reg.offset_GB + reg.length_GB)-1 );
        assert(reg.length_GB > 0);
      }
      else if(reg.length_GB > 0) {
        PINF(" - FREE: %lu (%lx-%lx)",
             reg.region_id,
             GB_to_bytes(reg.offset_GB),
             GB_to_bytes(reg.offset_GB + reg.length_GB)-1 );
      }
    }
  }

  void * get_region(uint64_t region_id, size_t * out_size) {
    for(uint16_t r=0;r<_region_count;r++) {
      auto reg = _regions[r];
      if(reg.region_id == region_id) {
        if(out_size)
          *out_size = (reg.length_GB << 30);
        return (reg.offset_GB  << 30) + arena_base();
      }
    }
    return nullptr; /* not found */
  }

  void erase_region(uint64_t region_id) {
    for(uint16_t r=0;r<_region_count;r++) {
      DM_region * reg = &_regions[r];
      if(reg->region_id == region_id) {
        reg->region_id = 0; /* power-fail atomic */
        mem_flush(&reg->region_id, sizeof(reg->region_id));
        return;
      }
    }
    throw API_exception("region (%lu) not found", region_id);
  }

  void * allocate_region(uint64_t region_id, unsigned size_in_GB) {
    for(uint16_t r=0;r<_region_count;r++) {
      auto reg = _regions[r];
      if(reg.region_id == region_id)
        throw API_exception("region_id already exists");
    }
    // TODO make crash-consistent
    uint32_t new_offset;
    bool found = false;
    for(uint16_t r=0;r<_region_count;r++) {
      DM_region * reg = &_regions[r];
      if(reg->region_id == 0  && reg->length_GB >= size_in_GB) {
        if(reg->length_GB == size_in_GB) {
          /* exact match */
          tx_atomic_write(reg, region_id);
          return (void*) ((((uint64_t)reg->offset_GB) << 30) + arena_base());
        }
        else {
          /* cut out */
          new_offset = reg->offset_GB;
          
          uint16_t changed_length = reg->length_GB - size_in_GB;
          uint16_t changed_offset = reg->offset_GB + size_in_GB;

          for(uint16_t r=0;r<_region_count;r++) {
            DM_region * reg_n = &_regions[r];
            if(reg_n->region_id == 0 && reg_n->length_GB == 0) {
              tx_atomic_write(reg, changed_offset, changed_length, 
                              reg_n, new_offset, size_in_GB, region_id);
              return (void*) ((((uint64_t)new_offset) << 30) + arena_base());
            }      
          }

        }
      }
    }
    if(!found)
      throw General_exception("no more regions (size in GB=%u)", size_in_GB);
    
    throw General_exception("no spare slots");
  }

  size_t get_max_available() const {
    size_t max_size = 0;
    for(uint16_t r=0;r<_region_count;r++) {
      const DM_region * reg = &_regions[r];
      if(reg->length_GB > max_size)
        max_size = reg->length_GB;
    }
    return GB_to_bytes(max_size);
  }
    

  inline uint64_t GB_to_bytes(unsigned GB) const {
    return ((uint64_t)GB) << 30;
  }
  
  inline void major_flush() {
    nupm::mem_flush(this, sizeof(DM_region_header) + (sizeof(DM_region) * _region_count));
  }
  
  bool check_magic() const {
    return (_magic == DM_REGION_MAGIC) && (_version == DM_REGION_VERSION);
  }

private:
  void tx_atomic_write(DM_region * dst, uint64_t region_id) {
    dst->region_id = region_id;
    mem_flush(&dst->region_id, sizeof(region_id));
  }

  void tx_atomic_write(DM_region * dst0, uint32_t offset0, uint32_t size0,
                       DM_region * dst1, uint32_t offset1, uint32_t size1, uint64_t region_id1) {
    /* TODO; */
    dst0->offset_GB = offset0;
    dst0->length_GB = size0;
    mem_flush(dst0, sizeof(DM_region));
    
    dst1->region_id = region_id1;
    dst1->offset_GB = offset1;
    dst1->length_GB = size1;
    mem_flush(dst1, sizeof(DM_region));

  }

  inline unsigned char * arena_base() { return (((unsigned char *)this) + GB(1)); }

  inline DM_region * region_table_base() { return _regions; }

  inline DM_region * region(size_t idx) {
    if(idx >= _region_count) return nullptr;
    DM_region * p = (DM_region *) region_table_base();
    return &p[idx];
  }                                                              
  
  void reset_header(size_t device_size) {
    _magic = DM_REGION_MAGIC;
    _version = DM_REGION_VERSION;
    _device_size = device_size;
    nupm::mem_flush(this, sizeof(DM_region_header));
  }
} __attribute__((packed));


}// namespace nupm

#endif //__NUPM_DAX_DATA_H__
