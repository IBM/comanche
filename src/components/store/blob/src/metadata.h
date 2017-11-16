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

#ifndef __BLOB_METADATA_H__
#define __BLOB_METADATA_H__

#include <string>
#include <string.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/spinlocks.h>
#include <tbb/tbb.h>
#include <core/dpdk.h>
#include <sqlite3.h>
#include <api/components.h>
#include <api/block_itf.h>

using namespace tbb;

struct __md_record
{
  struct {
    unsigned magic:      29;
    unsigned status:     2;
    unsigned block_size: 1;
  };
  uint32_t            crc;
  Common::Ticket_lock rlock;
  Common::Ticket_lock wlock;

  uint64_t start_lba;       
  uint64_t lba_count;
    
  unsigned char id[64];     
  unsigned char owner[64];  
  unsigned char datatype[32];
  unsigned char utc_modified[32]; // e.g. 2017-11-16T00:08:24+00:00
  unsigned char utc_created[32];  // e.g. 2017-11-16T00:08:24+00:00

} __attribute__((packed));

static constexpr unsigned MD_MAGIC = 1972;

enum {
  MD_STATUS_NOT_ASSIGN = 0,
  MD_STATUS_FREE = 1,
  MD_STATUS_USED = 2,
};

static_assert(sizeof(Common::Ticket_lock) == 4, "unexpected ticket lock size");
static_assert(sizeof(__md_record)==256,"md record size invalid");



/** 
 * Row-oriented in-memory representation of metadata. We use row-oriented because of
 * the impact on flushing.  This would be different for a persistent memory version
 * 
 */
class Metadata
{
private:
  static constexpr bool option_DEBUG = true;
  
public:
  /** 
   * Constructor
   * 
   * @param block_device 
   */
  Metadata(Component::IBlock_device * block_device, bool force_init = false);

  /** 
   * Destructor
   * 
   */
  ~Metadata();

private:
  bool check_validity();
  void initialize_space();
  void read_data(size_t start_lba, size_t count);
  void write_data(size_t start_lba, size_t count);
  
public:
  Component::IBlock_device * _block;
  Component::VOLUME_INFO     _vi;
  Component::io_buffer_t     _iob;
  __md_record *              _records;
  size_t                     _n_records;
  size_t                     _n_lba;
  
  tbb::concurrent_queue<__md_record*> _free_list;
};


#endif // __BLOB_METADATA_H__
