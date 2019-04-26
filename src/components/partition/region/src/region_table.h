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
#ifndef __REGION_TABLE_H__
#define __REGION_TABLE_H__

#include <common/utils.h>
#include <common/dump_utils.h>
#include <core/slab.h>
#include <core/avl_malloc.h>
#include <api/block_itf.h>


/** 
 * Region descriptor (128 bytes). region descriptors are not removed
 * but freed by setting'occupied' to false
 * 
 */
typedef struct
{
  addr_t  saddr;      // 8  ; starting storage address
  addr_t  vaddr;      // 16 ; virtual address that the client used for this region
  size_t  size;       // 24 ; size of region in bytes
  bool    occupied;   // 25 ; whether the region is occupied or not
  char    owner[20];  // 45 ; owner of the region
  char    id[83];     // 64 ; region identifier
} __attribute__((packed)) __region_desc_t;


/** 
 * Region table header (16 bytes)
 * 
 */
typedef struct  
{
  uint32_t magic;             // 4  ; magic number for sanity checking
  addr_t next;                // 12 ; addr of next table
  uint32_t num_entries;       // 16 ; number of entries in affinity_lba
  __region_desc_t entries[0]; // 16 ; region descriptors (see above)
} __attribute__((packed)) __region_table_t;


/** 
 * Class to help manager persistent region descriptor blocks
 * 
 */
class Region_table
{
private:
  static constexpr uint32_t REGION_MAGIC = 0x1DEA;
  static constexpr size_t   METADATA_FOOTPRINT = KB(4) * 1; //32;
  static constexpr addr_t   VADDR_BASE = 0x900000000ULL;
  static constexpr bool option_DEBUG = true;
  
public:
  
  Region_table(Component::IBlock_device * device,
               bool force_init)
    : _device(device)
  {
    device->get_volume_info(_vi);
    
    _io_buffer = device->allocate_io_buffer(METADATA_FOOTPRINT,
                                            _vi.block_size,/* alignment */
                                            Component::NUMA_NODE_ANY);

    
    assert(METADATA_FOOTPRINT % _vi.block_size == 0);
    _md_size_in_blocks = METADATA_FOOTPRINT / _vi.block_size;

    if(METADATA_FOOTPRINT % _vi.block_size)
      _md_size_in_blocks++;
    
    _table = static_cast<__region_table_t *>(device->virt_addr(_io_buffer));
      
    if(!_table) {
      throw Constructor_exception("Region_table::ctor - alloc_iomem- failed to allocate for region table");
    }
    
    if(force_init) {
      PLOG("Forced re-initialization..(max_entries=%ld)", max_entries());
      _table->magic = REGION_MAGIC;
      _table->num_entries = 0;
      _table->next = 0;
      flush();

      read_from_store();

      if(_table->magic != REGION_MAGIC) {
        throw Logic_exception("Second check Corrupt region table (magic=%x). Re-initializing..(max_entries=%ld)",
                              _table->magic, max_entries());
      }
    }
    else {
      PLOG("Reading existing region table from storage");
      read_from_store();
      if(_table->magic != REGION_MAGIC) {
        PNOTICE("Corrupt region table (magic=%x). Re-initializing..(max_entries=%ld)", _table->magic, max_entries());
        _table->magic = REGION_MAGIC;
        _table->num_entries = 0;
        _table->next = 0;
        flush();        
      }          
    }

    /* set up AVL range tree */
    _range_allocator = new Core::AVL_range_allocator(_volatile_slab,
                                                     METADATA_FOOTPRINT/KB(4),
                                                     _vi.block_count);
    PLOG("Region table: range allocator %lu-%lu", METADATA_FOOTPRINT/KB(4), _vi.block_count);
    assert(_vi.block_count > METADATA_FOOTPRINT/KB(4));
    
    size_t num_entries = _table->num_entries;
    PLOG("--- Regions (magic=%x) num_entries = %ld -------", _table->magic, num_entries);
    for(unsigned i=0;i<num_entries;i++) {
      if(_table->entries[i].occupied) {
        _range_allocator->alloc_at(_table->entries[i].saddr,
                                   _table->entries[i].size);
      }
    }

  }

  ~Region_table() {
    flush();
    assert(_table);
    _device->free_io_buffer(_io_buffer);

    delete _range_allocator;
  }

  static size_t max_entries() {
    return (METADATA_FOOTPRINT - sizeof(__region_table_t)) / sizeof(__region_desc_t);
  }

  size_t device_size_in_blocks() const {
    return _vi.block_count;
  }

  size_t num_entries() {
    return _table->num_entries;
  }

  size_t num_allocated_entries() {
    size_t n = 0;
    for(unsigned i=0;i<_table->num_entries;i++)
      if(_table->entries[i].occupied) n++;

    return n;
  }
  
  __region_desc_t * find(std::string owner, std::string id) {
    if(option_DEBUG && _table->num_entries == 0) {
      PLOG("no entries.");
      return nullptr;
    }
      
    for(unsigned i=0;i<_table->num_entries;i++) {
      if(option_DEBUG)
        PLOG("comparing against existing region in table (%s) against (%s)",
             _table->entries[i].id, id.c_str());
      if((owner.compare(_table->entries[i].owner)==0) &&
         (id.compare(_table->entries[i].id)==0)) {        
        return &_table->entries[i];
      }
    }
    return nullptr;
  }

  /** 
   * Allocate a new region
   * 
   * @param size Size of region in blocks
   * @param alignment Alignment of region in bytes 
   * @param owner Owner
   * @param id Region identifier
   * 
   * @return 
   */
  __region_desc_t * allocate(size_t size,
                             size_t alignment,
                             std::string owner,
                             std::string id) {
    assert(size > 0);
    Core::Memory_region* mr = _range_allocator->alloc(size,
                                                      alignment);
    assert(mr);
    return add_entry(owner, id, mr->addr(), vaddr_top(owner), size);
  }

  /** 
   * Get next virtual address entry
   * 
   * @param owner 
   * 
   * @return 
   */
  size_t vaddr_top(std::string owner) {
    size_t last_vaddr_top = VADDR_BASE;
    for(unsigned i=0;i<_table->num_entries;i++) {
      if(owner.compare(_table->entries[i].owner)==0) {
        addr_t this_top = _table->entries[i].vaddr +
          (_table->entries[i].size * _vi.block_size);
        
        if(this_top > last_vaddr_top)
          last_vaddr_top = this_top;
      }
    }
    return last_vaddr_top;
  }
  
  __region_desc_t * get_entry(size_t index) {
    return ((index + 1) > _table->num_entries) ? nullptr : &_table->entries[index];
  }

  bool remove_entry(std::string owner, std::string id)
  {
    __region_desc_t * rd = find(owner, id);
    if(rd == nullptr) return false;
    rd->occupied = false;
    flush();
    _range_allocator->free(rd->saddr);
    
    return true;
  }
  
  __region_desc_t* add_entry(std::string owner, std::string id, addr_t saddr, addr_t vaddr, size_t size) {
      
    /* check for existing free entry */
    size_t num_entries = _table->num_entries;
    for(unsigned i=0;i<num_entries;i++) {
      if(_table->entries[i].occupied == false) {
        PLOG("Reusing existing region descriptor");
        _table->entries[i].saddr = saddr;
        _table->entries[i].vaddr = vaddr;
        _table->entries[i].size = size;
        _table->entries[i].occupied = true;

        strncpy(_table->entries[i].owner, owner.c_str(), sizeof(_table->entries[i].owner)-1);
        _table->entries[i].owner[sizeof(_table->entries[i].owner)-1]='\0';
        strncpy(_table->entries[i].id, id.c_str(), sizeof(_table->entries[i].id)-1);
        _table->entries[i].id[sizeof(_table->entries[i].id)-1]='\0';
        flush();
        return &_table->entries[i];
      }           
    }
    /* else add a new entry */
    if(num_entries == max_entries())
      throw General_exception("region table full.");

    PLOG("Adding new entry");
    int i = num_entries;
    _table->entries[i].saddr = saddr;
    _table->entries[i].vaddr = vaddr;
    _table->entries[i].size = size;
    _table->entries[i].occupied = true;
    
    strncpy(_table->entries[i].owner, owner.c_str(), sizeof(_table->entries[i].owner)-1);
    _table->entries[i].owner[sizeof(_table->entries[i].owner)-1]='\0';
    strncpy(_table->entries[i].id, id.c_str(), sizeof(_table->entries[i].id)-1);
    _table->entries[i].id[sizeof(_table->entries[i].id)-1]='\0';

    _table->num_entries++;

    dump();
    flush();
    return &_table->entries[i];
  }

  addr_t dump() {
    size_t num_entries = _table->num_entries;
    PLOG("--- Region Table num_entries = %ld, magic = %x -------", num_entries, _table->magic);
    for(unsigned i=0;i<num_entries;i++) {
      PLOG("    region entry [%d]: 0x%lx va=0x%lx %ld %s %s %s",
           i,
           _table->entries[i].saddr,
           _table->entries[i].vaddr,
           _table->entries[i].size,
           _table->entries[i].occupied ? "-X-" : "-O-",
           _table->entries[i].owner,
           _table->entries[i].id
           );
    }
    PLOG("--------------------------------");
    return _table->next;
  }
  
  addr_t next_table() const {
    return _table->next;
  }
  
  void flush() {
    /* blind flush */
    PLOG("region_table flush: %ld blocks", _md_size_in_blocks);
    _device->write(_io_buffer, 0, 0, _md_size_in_blocks);
  }

  void read_from_store() {
    PLOG("region_table read from store: %ld blocks", _md_size_in_blocks);
    _device->read(_io_buffer, 0, 0, _md_size_in_blocks);
  }

private:

  Core::Slab::CRuntime<Core::Memory_region> _volatile_slab;
  Core::AVL_range_allocator *               _range_allocator;
  Component::VOLUME_INFO                    _vi;
  __region_table_t *                        _table;
  Component::IBlock_device *                _device;
  Component::io_buffer_t                    _io_buffer;
  size_t                                    _md_size_in_blocks;

};



#endif
