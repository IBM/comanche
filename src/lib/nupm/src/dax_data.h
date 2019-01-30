#ifndef __NUPM_DAX_DATA_H__
#define __NUPM_DAX_DATA_H__

#include <common/utils.h>
#include "pm_lowlevel.h"

#define DM_REGION_MAGIC 0xC0070000
#define DM_REGION_NAME_MAX_LEN 1024
#define DM_REGION_VERSION 1

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
  
  uint32_t  _magic;
  uint32_t  _version;
  uint64_t  _device_size;
  uint16_t  _region_count;  
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
        reg->region_id = 0;
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
          reg->region_id = region_id;
          mem_flush(&reg->region_id, sizeof(reg->region_id));
          return (void*) ((((uint64_t)reg->offset_GB) << 30) + arena_base());
        }
        else {
          /* cut out */
          reg->length_GB -= size_in_GB;
          new_offset = reg->offset_GB;
          reg->offset_GB += size_in_GB;
          found = true;
        }
      }
    }
    if(!found)
      throw General_exception("no more regions (size in GB=%u)", size_in_GB);
    
    for(uint16_t r=0;r<_region_count;r++) {
      DM_region * reg = &_regions[r];
      if(reg->region_id == 0 && reg->length_GB == 0) {
        reg->region_id = region_id;
        reg->offset_GB = new_offset;
        reg->length_GB = size_in_GB;
        major_flush();
        return (void*) ((((uint64_t)new_offset) << 30) + arena_base());
      }      
    }
    throw General_exception("no spare slots");
  }

  inline uint64_t GB_to_bytes(unsigned GB) {
    return ((uint64_t)GB) << 30;
  }
  
  inline void major_flush() {
    nupm::mem_flush(this, sizeof(DM_region_header) + (sizeof(DM_region) * _region_count));
  }
  
  bool check_magic() {
    return (_magic == DM_REGION_MAGIC) && (_version == DM_REGION_VERSION);
  }

private:
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
