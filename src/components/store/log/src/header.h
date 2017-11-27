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
#include <common/cycles.h>
#include <api/block_itf.h>

class Header
{
  static constexpr uint32_t MAGIC = 0xAC1DBA5E;

  struct Store_master_block
  {
    uint32_t magic;
    uint32_t flags;
    lba_t    next_free_lba;
    lba_t    max_lba;
    uint64_t fixed_size;
    char     owner[512];
    char     name[512];
    index_t  tail;
    uint64_t timestamp;
  } __attribute__((packed));

public:
  Header(std::string owner,
         std::string name,
         Component::IBlock_device * block,
         uint64_t fixed_size,
         bool force_init) : _block(block)
  {
    if(block == nullptr)
      throw API_exception("bad Header constructor parameters");

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
      PLOG("Log-store: header magic mismatch, reinitializing");
      force_init = true;
    }

    if(force_init) {
      PLOG("Log-store: forced initialization");
      memset(_mb, 0, _vi.block_size); /* zero whole first sector */
      _mb->magic = MAGIC;
      _mb->max_lba = _vi.max_lba;
      _mb->fixed_size = fixed_size;
      _mb->next_free_lba = 1; /* block 0 is used by the header */
      _mb->tail = 0;
      strncpy(_mb->name, name.c_str(), 511);
      strncpy(_mb->owner, owner.c_str(), 511);      
      write_mb();
    }
    else {
      PLOG("Log-store: using existing state");
    }
    PLOG("Log-store: next=%ld max=%ld owner=%s name=%s",
         _mb->next_free_lba, _mb->max_lba, _mb->owner, _mb->name);

    if(name.compare(_mb->name) != 0)
      throw General_exception("Log-store: name does not match");

    if(owner.compare(_mb->owner) != 0)
      throw General_exception("Log-store: owner does not match");

    if(fixed_size != _mb->fixed_size)
      throw General_exception("Log-store: fixed size mismatch");
         
  }

  ~Header() {
    _block->free_io_buffer(_iob);
    _block->release_ref();
  }

  void dump_info() {
    std::lock_guard<std::mutex> g(_lock);
    PINF("Log-Store");
    PINF("Header: magic=0x%x", _mb->magic);
    PINF("      : flags=0x%x", _mb->flags);
    PINF("      : timestamp %lx", _mb->timestamp);
    PINF("      : next_free_lba=%lu", _mb->next_free_lba);
    PINF("      : owner=%s", _mb->owner);
    PINF("      : name=%s", _mb->name);
    PINF("      : tail=%lu", _mb->tail);
    PINF("      : fixed_size=%lu", _mb->fixed_size);
    PINF("      : used blocks %lu / %lu (%f %%)",
         _mb->next_free_lba, _mb->max_lba,
         (((float)_mb->next_free_lba)/((float)_mb->max_lba))*100.0);
    PINF("      : used capacity %lu MB", REDUCE_MB(_mb->next_free_lba * _vi.block_size));
    PINF("      : record count %lu (including tail excess) ", _mb->tail / _mb->fixed_size);
  }
      
  void flush() {
    write_mb();
  }

  index_t& get_tail() const { return _mb->tail; }
  
  size_t block_size() const { return _vi.block_size; }
  
  lba_t allocate(size_t n_bytes, size_t& n_blocks, index_t& index) {

    if(unlikely(n_bytes % _vi.block_size))
      throw API_exception("allocate must round to block size");

    n_blocks = n_bytes / _vi.block_size;   

    lba_t result;

    if((_mb->max_lba - _mb->next_free_lba) < n_blocks)
      throw API_exception("Log-store: no more blocks");
    result = _mb->next_free_lba;
    _mb->next_free_lba += n_blocks;
    index = _mb->tail;
    _mb->tail += n_bytes;
    return result;
  }

private:

  void write_mb() {
    PLOG("Log-store: writing out header");
    std::lock_guard<std::mutex> g(_lock);
    _mb->timestamp = rdtsc(); /* update timestamp */
    _block->write(_iob,0,0/*lba*/,1);
  }

  void read_mb() {
    std::lock_guard<std::mutex> g(_lock);
    _block->read(_iob,0,0/*lba*/,1);
  }

private:
  Component::IBlock_device * _block;
  Component::VOLUME_INFO     _vi;
  Component::io_buffer_t     _iob;
  Store_master_block *       _mb;
  std::mutex                 _lock;

};


#endif
