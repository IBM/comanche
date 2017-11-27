#include <iostream>
#include <api/metadata_itf.h>
#include "metadata.h"
#include "md_record.h"

using namespace Component;

Metadata::Metadata(Component::IBlock_device * block_device,
                   unsigned block_size,
                   int flags)
{
  if(!block_device || !block_size)
    throw Constructor_exception("Metadata::ctor bad param");
  
  _block = block_device;
  _block->add_ref();

  _block->get_volume_info(_vi);
  _n_records = (_vi.max_lba + 1) * (_vi.block_size / sizeof(struct __md_record));
  assert(_vi.block_size % sizeof(struct __md_record) == 0);

  _memory_size = (_vi.max_lba + 1) * (_vi.block_size);
  
  if(option_DEBUG)
    PLOG("Metadata: #records = %lu (%lu KiB)",
         _n_records,
         REDUCE_KB(_memory_size));

  
  /* create IO buffer */
  _iob = _block->allocate_io_buffer(_memory_size, KB(4), NUMA_NODE_ANY);
  if(option_DEBUG)
    PLOG("Metadata: allocated IO buffer (%p)", _block->virt_addr(_iob));

  /* read in all metadata */
  _block->read(_iob,
               0, // offset
               0, // lba
               _vi.max_lba + 1); // lba count

  _records = static_cast<struct __md_record *>(_block->virt_addr(_iob));

  if(!scan_records_unsafe() || (flags & FLAGS_FORMAT)) {
    PLOG("initializing metadata storage");
    reinitialize(); /* TODO: partial rebuild */
  }

  _lock_array = new Lock_array(_n_records);

  if(option_DEBUG)
    dump_info();
}

Metadata::~Metadata()
{
  delete _lock_array;
  
  _block->free_io_buffer(_iob);
  _block->release_ref();  
}

void Metadata::allocate(uint64_t start_lba,
                        uint64_t lba_count,
                        const char * id,
                        const char * owner,
                        const char * datatype)
{
  apply([=](struct __md_record& record, size_t index, bool& keep_going)
        {
          lock(index);

          if(record.is_free()) {
            record.set_used();
            record.start_lba = start_lba;
            record.lba_count = lba_count;
            record.set_id(id);
            record.set_owner(owner);
            record.set_datatype(datatype);
            keep_going = false;
            flush_record(&record);
          }
          unlock(index);
        }
        , 0, _n_records);

}

void Metadata::flush_record(struct __md_record * record)
{
  assert(record >= _records);

  addr_t block_addr = (round_down((addr_t) record, _vi.block_size));
  PLOG("record_addr=%p block_addr=%lx", record, block_addr);
  
  addr_t lba = (block_addr - (addr_t)_records) / _vi.block_size;  
  if(option_DEBUG) {    
    PLOG("flushing %p (in block 0x%lx)", record, lba);
  }
  _block->write(_iob,
                (addr_t)_records - block_addr, // offset
                lba,
                1); // lba_count
                
}

void Metadata::reinitialize()
{
  assert(_vi.block_size == 512 ||
         _vi.block_size == 4096);
  
  apply([=](struct __md_record& record, size_t index, bool& keep_going)
        {
          record.clear();
          if(_vi.block_size == 512)
            record.block_size = MD_BLOCK_SIZE_512;
          else
            record.block_size = MD_BLOCK_SIZE_4096;
        }
        , 0, _n_records);
  
  _block->write(_iob,
                0, // offset
                0, // lba
                _vi.max_lba + 1); // lba count  
}

bool Metadata::scan_records_unsafe()
{
  for(unsigned long i=0;i<_n_records;i++) {
    if(!_records[i].check_magic()) {
      if(option_DEBUG)
        PWRN("bad magic: record %lu", i);
      return false;
    }
  }
  return true;
}

void Metadata::dump_info()
{
  unsigned long free = 0;
  unsigned long used = 0;
  unsigned long unassigned = 0;
  
  apply([=,&free, &used, &unassigned](struct __md_record& record,
                                    size_t index,
                                    bool& keep_going)
        {
          lock(index);
          
          if(record.is_used())
            used ++;
          else if(record.is_free())
            free ++;
          else
            unassigned ++;

          if(record.is_used())
            record.dump_record();

          unlock(index);
        }
        , 0, _n_records);
  PINF("== Metadata Summary ==");
  PINF("used:       %lu", used);
  PINF("free:       %lu", free);
  PINF("unassigned: %lu", unassigned);
}

void Metadata::apply(std::function<void(struct __md_record&,
                                        size_t index,
                                        bool& keep_going)> func,
                     unsigned long start,
                     unsigned long count)
{
  if((start + count) > _n_records)
    throw API_exception("out of bounds");

  bool keep_going = true;
  for(unsigned long i=start;i<count;i++) {
    func(_records[i],i, keep_going);
    if(!keep_going)
      break;
  }
}

size_t Metadata::get_record_count()
{
  return 0;
}

IMetadata::iterator_t Metadata::open_iterator(std::string filter)
{
  return nullptr;
}

status_t Metadata::iterator_get(IMetadata::iterator_t iter,
                                std::string& out_metadata,
                                void *& allocator_handle,
                                uint64_t* lba,
                                uint64_t* lba_count)
{
  return E_FAIL;
}

size_t Metadata::iterator_record_count(iterator_t iter)
{
  return 0;
}

void Metadata::close_iterator(iterator_t iterator)
{
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Metadata_factory::component_id()) {
    return static_cast<void*>(new Metadata_factory());
  }
  else return NULL;
}

