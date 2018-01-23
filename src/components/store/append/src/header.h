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

#ifndef __STORE_HEADER_H__
#define __STORE_HEADER_H__

#include <mutex>
#include <api/block_itf.h>

class Header
{
  static constexpr uint32_t MAGIC = 0xACDCEEEE;

  struct Store_master_block
  {
    uint32_t magic;
    uint32_t flags;
    lba_t    next_free_lba;
    lba_t    block_count;
    char     owner[512];
    char     name[512];
  } __attribute__((packed));

public:
  Header(Component::IBlock_device * block,
         std::string owner,
         std::string name,
         bool force_init) : _block(block) {
    using namespace Component;
    assert(block);
    block->add_ref();
    block->get_volume_info(_vi);
    assert(_vi.block_size == 4096 || _vi.block_size == 512);
    _iob = block->allocate_io_buffer(_vi.block_size, _vi.block_size, NUMA_NODE_ANY);
    assert(sizeof(Store_master_block) <= _vi.block_size);
    _mb = static_cast<Store_master_block*>(block->virt_addr(_iob));
    assert(_mb);

    read_mb();
    if(_mb->magic != MAGIC) {
      PLOG("Append-store: header magic mismatch, reinitializing");
      force_init = true;
    }
    else if(force_init) {
      PLOG("Append-store: forced init");
    }

    if(force_init) {
      memset(_mb, 0, _vi.block_size); /* zero whole first sector */
      _mb->magic = MAGIC;
      _mb->block_count = _vi.block_count;
      _mb->next_free_lba = 1;
      strncpy(_mb->name, name.c_str(), 511);
      strncpy(_mb->owner, owner.c_str(), 511);      
      write_mb();
    }
    else {
      PLOG("Append-store: using existing state");
      _existing = true;
    }
    PLOG("Append-store: next=%ld max=%ld owner=%s name=%s",
         _mb->next_free_lba, _mb->block_count, _mb->owner, _mb->name);

    if(name.compare(_mb->name) != 0)
      PLOG("Append-store: name does not match");

    if(owner.compare(_mb->owner) != 0)
      PLOG("Append-store: owner does not match");
    
         
  }

  ~Header() {
    dump_info();
    write_mb();
    _block->free_io_buffer(_iob);
  }

  void dump_info() {
    std::lock_guard<std::mutex> g(_lock);
    PINF("Append-Store");
    PINF("Header: magic=0x%x", _mb->magic);
    PINF("      : flags=0x%x", _mb->flags);
    PINF("      : next_free_lba=%lu", _mb->next_free_lba);
    PINF("      : owner=%s", _mb->owner);
    PINF("      : name=%s", _mb->name);
    PINF("      : used blocks %lu / %lu (%f %%)",
         _mb->next_free_lba, _mb->block_count,
         (((float)_mb->next_free_lba)/((float)_mb->block_count))*100.0);
  }
      
  void flush() {
    write_mb();
  }

  lba_t allocate(size_t n_bytes, size_t& n_blocks) {

    assert(n_bytes > 0);
    /* smallest multiple of blocksize >= nbytes */
    n_blocks = (n_bytes + _vi.block_size - 1) / _vi.block_size; 

    lba_t result;
    {
      std::lock_guard<std::mutex> g(_lock);
      if((_mb->block_count - _mb->next_free_lba) < n_blocks)
        throw API_exception("Append-store: no more blocks");
      result = _mb->next_free_lba;
      _mb->next_free_lba += n_blocks;
    }
    return result;
  }

  bool existing() const { return _existing; }
  
private:

  void write_mb() {
    _block->write(_iob,0,0,1);
  }

  void read_mb() {
    _block->read(_iob,0,0,1);
  }

private:
  Component::IBlock_device * _block;
  Component::VOLUME_INFO     _vi;
  Component::io_buffer_t     _iob;
  Store_master_block *       _mb;
  std::mutex                 _lock;
  bool                       _existing = false;
};


#endif
