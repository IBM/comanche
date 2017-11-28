#include <sstream>
#include <core/physical_memory.h>
#include <common/logging.h>
#include "log_store.h"
#include "buffer_manager.h"

/** 
 * Buffer management pattern
 * - allocate single IO buffer and partition
 * - managed partitioned indices in FIFO
 */
using namespace Component;

Log_store::Log_store(std::string owner,
                     std::string name,
                     Component::IBlock_device* block,
                     int flags,
                     size_t fixed_size,
                     bool use_crc)
  : _hdr(owner, name, block, fixed_size, flags & FLAGS_FORMAT),
    _use_crc(use_crc),
    _bm(block, 0, _hdr),
    _fixed_size(fixed_size)
{
  int rc;
  
  if(block == nullptr)
    throw API_exception("bad Log_store constructor parameters");

  if(fixed_size == 0)
    throw API_exception("variable size log items not implemented");

  /* open block device */
  _lower_layer = block;
  _lower_layer->add_ref();

  block->get_volume_info(_vi);
  PLOG("Log-store: block device capacity=%lu",
       _vi.max_lba);

  _max_io_blocks = _vi.max_dma_len / _vi.block_size;
  _max_io_bytes  = _vi.max_dma_len;
  assert(_vi.max_dma_len % _vi.block_size == 0);

  assert(_vi.max_dma_len == 0 || Buffer_manager::IO_BUFFER_SIZE <= _vi.max_dma_len);

  /* allocate buffer */
  _iob = _lower_layer->allocate_io_buffer(round_up(_fixed_size,_vi.block_size)
                                          + (_vi.block_size*2),
                                          KB(4),
                                          Component::NUMA_NODE_ANY);

}

Log_store::~Log_store()
{
  _lower_layer->free_io_buffer(_iob);
  _lower_layer->release_ref();
}


index_t Log_store::write(const void * data,
                         const size_t data_len,
                         unsigned queue_id)
{
  index_t index;
  
  /* write length */
  if(data_len > INT32_MAX)
    throw API_exception("length too large for 32bit representation");

  if(_fixed_size > 0 && data_len != _fixed_size)
    throw API_exception("mismatched size in write call (expect=%ld request=%ld)", _fixed_size, data_len);

  if(option_DEBUG)
    PLOG("Log_store: write %s", (char*)data);
  
  uint32_t crc;

  if(_use_crc)
    crc = crc32(0UL, (const Bytef*) data, data_len); /* don't hold lock doing this */

  {
    std::lock_guard<std::mutex> g(_lock);

    if(_use_crc) {
      if(_fixed_size > 0) {
        /* write crc and data */
        index = _bm.write_out(crc, queue_id);
        _bm.write_out(data, data_len, queue_id);
        return index / _fixed_size; /*< return record index */
      }
      else {
        /* write len, crc, data */
        index = _bm.write_out(data_len, queue_id);
        _bm.write_out(crc, queue_id);
        _bm.write_out(data, data_len, queue_id);
        return index;
      }
    }
    else { /* no CRC */
      if(_fixed_size > 0) {
        /* write data only */
        index = _bm.write_out(data, data_len, queue_id);
        return index / _fixed_size; /*< return record index */
      }
      else {
        /* len + data */
        index = _bm.write_out(data_len, queue_id);
        _bm.write_out(data, data_len, queue_id);
        return index;
      }
    }    
  }
}


byte * Log_store::read(const index_t index,
                       Component::io_buffer_t iob,
                       unsigned queue_id)
{
  addr_t record_pos;
  if(_fixed_size) record_pos = index * _fixed_size;    
  else throw API_exception("read on non-fixed size not implemented");

  auto required_size = _fixed_size + _vi.block_size;
    
  if(_lower_layer->get_size(iob) < required_size)
    throw API_exception("insufficiently sized buffer in Log_store::read call. len %ld bytes required", required_size);

  unsigned blocks_to_read = 0;
  unsigned bottom_lba = round_down(record_pos, _vi.block_size) / _vi.block_size;
  unsigned top_lba = round_up(record_pos + _fixed_size, _vi.block_size) / _vi.block_size;
  unsigned total_blocks = top_lba - bottom_lba + 1;
  unsigned offset_in_lba = record_pos % _vi.block_size;

  if(option_DEBUG)
    PLOG("bottom_lba=%u, top_lba=%u, total_blocks=%u offset=%u",
         bottom_lba+1, top_lba+1, total_blocks, offset_in_lba);

  
  _lower_layer->read(iob,
                     0, /* offset in IOB */
                     bottom_lba + 1, /* add one block because of header */
                     total_blocks,
                     queue_id);
  
  byte * vaddr = static_cast<byte*>(_lower_layer->virt_addr(iob));
  return vaddr + offset_in_lba;
}

std::string Log_store::read(const index_t index)
{
  char * ptr = (char*) this->read(index, _iob, 0);
  std::string result(ptr);
  return result;
}

status_t Log_store::flush(unsigned queue_id)
{
  _bm.flush_buffer();
  _hdr.flush(); /* flush metadata */
  _lower_layer->check_completion(0, queue_id); /* wait for all pending */
  return S_OK;
}

void Log_store::dump_info()
{
  _bm.dump_info();
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Log_store_factory::component_id()) {
    return static_cast<void*>(new Log_store_factory());
  }
  else return NULL;
}
