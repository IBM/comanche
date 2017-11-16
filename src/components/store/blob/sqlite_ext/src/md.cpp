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

#include <string>
#include <string.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <core/dpdk.h>
#include <sqlite3.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <api/pmem_itf.h>
#include <api/partition_itf.h>

#include "md.h"

/** 
 * Metadata residing on block device using SQLite3 as the front end.  We use the
 * SQLite3 virtual table mechanism.
 * 
 */

#define BLOCK_SIZE 4096
#define SCHEMA "CREATE TABLE x(id, ns, owner, type, created, modified, start_lba, nblocks, wlock, rlock)";


using namespace Component;

enum {
  MDDB_RECORD_STATUS_FREE = 1,
  MDDB_RECORD_STATUS_USED = 2,
  MDDB_RECORD_STATUS_DELETED = 3,
};

/** 
 * Each entry is 512 bytes, 8 entries to a 4K storage block.  We can
 * play with these fields later as we see fit.
 * 
 */
struct __mddb_record
{
  uint32_t magic;
  uint32_t crc;
  uint8_t  status; /* set to 1 if used */
  uint8_t  rlock;
  uint8_t  wlock;
  uint8_t  block_size;
  uint64_t start_lba;
  uint64_t lba_count; // 28 bytes
  
  unsigned char id[64];
  unsigned char owner[64];

  unsigned char datatype[40];
  unsigned char utc_modified[32]; // e.g. 2017-11-16T00:08:24+00:00
  unsigned char utc_created[32]; // e.g. 2017-11-16T00:08:24+00:00
                   
} __attribute__((packed));


static_assert(sizeof(__mddb_record)==512,"header size invalid");




class Metadata : public sqlite3_vtab /* subclass sqlite3_vtab is very important */
{
private:
  static constexpr bool option_DEBUG = true;

public:

  Metadata(const char * block_device_pci,
           unsigned partition,
           const char * owner,
           int core);
  ~Metadata();
  char * get_schema();
  void shutdown();
  inline void check_canary() { assert(_canary == 0xF00D); }

private:

  uint32_t              _canary;
  IBlock_device *       _block_base;
  IPartitioned_device * _gpt;
  IBlock_device *       _block;
  IPersistent_memory *  _pmem;
  VOLUME_INFO           _block_base_vi;
  std::string           _schema;
  unsigned long         _record_capacity;
};

    

/** 
 * Metadata class
 * 
 */

Metadata::Metadata(const char * block_device_pci,
                   unsigned partition,
                   const char * owner, int core) :
  _canary(0xF00D)
{
  if(option_DEBUG) {
    PLOG("::Metadata(%s,%u) this->%p", block_device_pci, partition, this);
    PLOG("Canary (%x) @ %p", _canary, &_canary);
  }

  DPDK::eal_init(8,0,true);

  /* create block device */
  {
    cpu_mask_t cpus;
    cpus.add_core(core);

    IBase * comp = load_component("libcomanche-blknvme.so",
                                  block_nvme_factory);

    IBlock_device_factory * fact = (IBlock_device_factory *)
      comp->query_interface(IBlock_device_factory::iid());

    _block_base = fact->create(block_device_pci, &cpus);
    assert(_block_base);
    _block_base->get_volume_info(_block_base_vi);

    if(_block_base_vi.block_size != 4096)
      throw Constructor_exception("only 4K lbaf supported");    
    
    fact->release_ref();
    if(option_DEBUG) PMAJOR("NVMe block device created OK. (%p)", _block_base);
  }



  /* create partition overlay */
  {
    IBase * comp = load_component("libcomanche-partgpt.so",
                                  part_gpt_factory);
    IPartitioned_device_factory* fact = (IPartitioned_device_factory *)
      comp->query_interface(IPartitioned_device_factory::iid());
    assert(fact);

    _gpt = fact->create(_block_base); /* pass in lower-level block device */
    fact->release_ref();
    if(option_DEBUG) PMAJOR("GUID partition component created OK (%p).", _gpt);

    _block = _gpt->open_partition(partition);
    assert(_block);

    VOLUME_INFO vi;
    _block->get_volume_info(vi);
    assert(vi.block_size == 4096);

    _record_capacity = ((vi.max_lba + 1) * 4069)/sizeof(__mddb_record);
    
    if(option_DEBUG) PMAJOR("Opened partition OK. blocks=%lu record_capacity=%lu (%p)",
                            vi.max_lba,
                            _record_capacity,
                            _block);
  }

  /* create persistent memory */
  {
    IBase * comp = load_component("libcomanche-pmemfixed.so",
                                  pmem_fixed_factory);

    IPersistent_memory_factory * fact = static_cast<IPersistent_memory_factory *>
      (comp->query_interface(IPersistent_memory_factory::iid()));
    assert(fact);
    _pmem = fact->open_allocator(owner, _block, false /* force init */);
    assert(_pmem);

    fact->release_ref();

    if(option_DEBUG) PMAJOR("Persistent memory component created OK (%p).", _pmem);
    _pmem->start();
  }

  _schema = SCHEMA;

  check_canary();
}

Metadata::~Metadata()
{
}

void Metadata::shutdown()
{
  check_canary();
  
  _pmem->release_ref();
  _block->release_ref();
  _gpt->release_ref();
  _block_base->release_ref();  
}

char * Metadata::get_schema()
{
  check_canary();
  char * schema = (char *) sqlite3_malloc(_schema.size() + 1);
  strcpy(schema, _schema.c_str());
  return schema;
}



/** 
 * C wrappers for Metadata class
 * 
 */
extern "C" {
  
  void * mddb_create_instance(const char * block_device_pci,
                              unsigned partition,
                              const char * owner,
                              int core) {
    return ::new Metadata(block_device_pci, partition, owner, core);
  }

  void mddb_check_canary(void * _this) {
    static_cast<Metadata *>(_this)->check_canary();
  }
  
  void mddb_free_instance(void * _this) {
    static_cast<Metadata *>(_this)->shutdown();
    delete static_cast<Metadata *>(_this);
  }

  // void mddb_foo(void * _this) {
  //   static_cast<Metadata *>(_this)->foo();
  // }

  char * mddb_get_schema(void * _this) {
    return static_cast<Metadata *>(_this)->get_schema();
  }
}
