#include <iostream>
#include <regex>
#include <sstream>
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <api/metadata_itf.h>
#include <common/dump_utils.h>
#include "metadata.h"
#include "md_record.h"

using namespace Component;
using namespace rapidjson;

Metadata::Metadata(Component::IBlock_device * block_device,
                   unsigned block_size,
                   int flags) : _block_size(block_size)
{
  if(!block_device || !block_size)
    throw Constructor_exception("Metadata::ctor bad param");
  
  _block = block_device;
  _block->add_ref();

  _block->get_volume_info(_vi);

  if(option_DEBUG) {
    PLOG("Metadata: managed block_size = %u", block_size);
    PLOG("Metadata: underlying volume block count = %ld blocks", _vi.block_count);
  }
  
  _n_records = _vi.block_count * (_vi.block_size / sizeof(struct __md_record));
  assert(_vi.block_size % sizeof(struct __md_record) == 0);

  _memory_size = _vi.block_count * _vi.block_size;
  
  if(option_DEBUG)
    PLOG("Metadata: #records = %lu (%lu MiB)",
         _n_records,
         REDUCE_MB(_memory_size));

  
  /* create IO buffer */
  _iob = _block->allocate_io_buffer(_memory_size, KB(4), NUMA_NODE_ANY);
  _records = static_cast<struct __md_record *>(_block->virt_addr(_iob));
  memset(_records,0x1,_memory_size);
  
  if(option_DEBUG)
    PLOG("Metadata: allocated IO buffer (%p) of %lu bytes", _block->virt_addr(_iob), _memory_size);

  /* read in all metadata */
  _block->read(_iob,
               0, // offset
               0, // lba
               _vi.block_count); // lba count

  //  hexdump(_records,512);
  _lock_array = new Lock_array(_n_records);

  if(scan_records_unsafe()==false || (flags & FLAGS_FORMAT)) {
    PLOG("wipe-initializing metadata storage");
    wipe_initialize(); /* TODO: partial rebuild */
  }
  else {
    initialize();
  }

  memset(_records,0x1,_memory_size);
  _block->read(_iob,
               0, // offset
               0, // lba
               _vi.block_count); // lba count

  assert(scan_records_unsafe());

  if(option_DEBUG)
    dump_info();
}

Metadata::~Metadata()
{
  delete _lock_array;
  
  _block->free_io_buffer(_iob);
  _block->release_ref();  
}

index_t Metadata::allocate(uint64_t start_lba,
                           uint64_t lba_count,
                           const std::string& id,
                           const std::string& owner,
                           const std::string& datatype)
{
  struct __md_record* precord;
  if(!_free_list.try_pop(precord)) {
    return -E_EMPTY;
  }

  lock(precord->index);

  precord->block_size = _block_size;
  precord->set_used();
  precord->start_lba = start_lba;
  precord->lba_count = lba_count;
  precord->set_id(id.c_str());
  precord->set_owner(owner.c_str());
  precord->set_datatype(datatype.c_str());

  unlock(precord->index);

  flush_record(precord);

  return precord->index;
}

void Metadata::flush_record(__md_record * record)
{
  assert(record >= _records);

  addr_t block_addr = (round_down((addr_t) record, _vi.block_size));
  PLOG("flushing: record_addr=%p block_addr=%lx", record, block_addr);
  
  addr_t lba = (block_addr - (addr_t)_records) / _vi.block_size;  
  if(option_DEBUG) {    
    PLOG("flushing %p (in block 0x%lx at offset %lu)", record, lba, block_addr - ((addr_t)_records));
  }

  /* assumes records do not straddle block boundaries */
  __md_record * record_batch_base = (struct __md_record *) block_addr;

  /* grab all the locks for records in the same block */
  size_t records_per_block = _vi.block_size / sizeof(__md_record);
  for(unsigned i=0;i<records_per_block;i++)
    lock(record_batch_base[i].index);
  
  _block->write(_iob,
                block_addr - ((addr_t)_records), // offset
                lba,
                1); // lba_count

  for(unsigned i=0;i<records_per_block;i++)
    unlock(record_batch_base[i].index);

}

void Metadata::wipe_initialize()
{
  assert(_vi.block_size == 512 ||
         _vi.block_size == 4096);
  
  apply([=](struct __md_record& record, size_t index, bool& keep_going)
        {
          record.clear(); /* this will set magic too */
          record.block_size = (_block_size == 512) ? MD_BLOCK_SIZE_512 : MD_BLOCK_SIZE_4096;
          record.index = index;

          _free_list.push(&record);
        }
        , 0, _n_records);
  
  _block->write(_iob,
                0, // offset
                0, // lba
                _vi.block_count); // lba count
}

void Metadata::initialize()
{
  assert(_vi.block_size == 512 ||
         _vi.block_size == 4096);
  
  apply([=](struct __md_record& record, size_t index, bool& keep_going)
        {
          if(record.is_free()) {
            _free_list.push(&record);
          }          
        }
        , 0, _n_records);
  
  _block->write(_iob,
                0, // offset
                0, // lba
                _vi.block_count); // lba count  
}



bool Metadata::scan_records_unsafe()
{
  for(unsigned long i=0;i<_n_records;i++) {
    if(!_records[i].check_magic()) {
      if(option_DEBUG)
        PWRN("scan detected bad magic (%u): record %lu", _records[i].magic, i);
      return false;
    }
  }
  if(option_DEBUG)
    PLOG("Metadata: integrity scan OK!");
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
  assert(_free_list.unsafe_size() <= _n_records);
  return _n_records - _free_list.unsafe_size();
}

struct __iterator_t
{
  size_t     pos;
  std::regex id_filter;
  std::regex owner_filter;
};

IMetadata::iterator_t Metadata::open_iterator(std::string filter)
{
  Document doc;
  __iterator_t * i;
  try {
    doc.Parse(filter.c_str());

    if(option_DEBUG)
      PLOG("filter expr (%s)", filter.c_str());

    if(doc.IsNull())
      throw API_exception("fixobd: invalid filter string");

    i = new __iterator_t;
    i->pos = 0;
    
    if(doc.HasMember("id") && doc["id"].IsString()) {
      i->id_filter = std::regex(doc["id"].GetString());

      if(option_DEBUG)
        PLOG("id_filter (%s)", doc["id"].GetString());
    }
    
  }
  catch(...) {
    throw API_exception("invalid filter string");
  }

  return static_cast<void*>(i);
}

status_t Metadata::iterator_get(IMetadata::iterator_t iter,
                                index_t& out_index,
                                std::string& out_metadata,
                                uint64_t* lba,
                                uint64_t* lba_count)
{
  __iterator_t * i = static_cast<__iterator_t*>(iter);

  do {
    if(i->pos >= _n_records) {
      break;
    }

    auto& record = _records[i->pos];

    i->pos ++;
    if(record.is_free()) continue;
        
    std::smatch id_match, owner_match;
    const std::string id_target = record.id;
    const std::string owner_target = record.owner;

    if(std::regex_match(id_target, id_match, i->id_filter) ||
       std::regex_match(owner_target, owner_match, i->owner_filter)) {
      /* create JSON result */
      std::stringstream ss;
      ss << "{\"id\": \"" << record.id << "\",\"owner\":\"" << record.owner << "\","
         << "\"datatype\":\"" << record.datatype << "\",\"utc_modified\":\"" << record.utc_modified
         << "\",\"utc_created\":\"" << record.utc_created << "\"}";
      
      out_metadata = ss.str();
      out_index = i->pos - 1;
      
      if(lba) *lba = record.start_lba;
      if(lba_count) *lba_count = record.lba_count;
      return S_OK;
    }
  }
  while(i->pos < _n_records);
  
  return E_EMPTY;
}

bool Metadata::check_exists(const std::string& id, const std::string& owner, size_t& out_size)
{
  bool check_owner = owner.empty() ? false : true;
  
  /* this is an unsafe scan! */
  for(unsigned long i=0;i<_n_records;i++) {
    auto& record = _records[i];
    if(id.compare(record.id) == 0) {
      if(!check_owner || owner.compare(record.owner) == 0) {
        out_size = record.block_size ? (record.lba_count * 512) : (record.lba_count * 4096);
        PLOG("Metadata: returning size=%lu", out_size);
        return true;
      }
    }
  }
  return false;
}


std::string Metadata::get_metadata(index_t index)
{
  if(index > _n_records)
    throw API_exception("invalid index parameter");

  lock(index);
  auto& record = _records[index];

  std::stringstream ss;
  ss << record.id << " ";
  ss << record.owner << " ";
  ss << record.datatype << " ";
  ss << (record.lba_count * _vi.block_size) << " ";
   
  unlock(index);
  return ss.str();
}

void Metadata::close_iterator(iterator_t iter)
{
  __iterator_t * i = static_cast<__iterator_t*>(iter);
  assert(i);
  delete i;
}

void Metadata::free(index_t index)
{
  if(index > _n_records)
    throw API_exception("invalid parameter");

  struct __md_record* record = &_records[index];

  lock(index);

  if(record->is_free())
    throw API_exception("invalid parameter; record already free");

  record->set_free();
  _free_list.push(record); /* ok, list is thread safe */
 
  unlock(index);

  flush_record(record); /* update persist copy */
}

  
void Metadata::lock_entry(index_t index)
{
  if(index > _n_records)
    throw API_exception("invalid parameter");
  lock(index);
}

void Metadata::unlock_entry(index_t index)
{
  if(index > _n_records)
    throw API_exception("invalid parameter");
  unlock(index);
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

