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

#ifndef __APPEND_STORE_H__
#define __APPEND_STORE_H__

#include <sqlite3.h>
#include <string>
#include <thread>
#include <core/zerocopy_passthrough.h>
#include <api/store_itf.h>
#include <api/region_itf.h>
#include <api/block_allocator_itf.h>

#include "header.h"

class Append_store : public Core::Zerocopy_passthrough_impl<Component::IStore>
{  
private:
  static constexpr unsigned DMA_ALIGNMENT_BYTES = 8;
  static constexpr bool option_DEBUG = false;
  static constexpr bool option_STATS = false;
public:

  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  Append_store(const std::string owner,
               const std::string name,
               const std::string db_location,
               Component::IBlock_device* block,
               int flags);

  /** 
   * Destructor
   * 
   */
  virtual ~Append_store();
 

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  
  DECLARE_COMPONENT_UUID(0x679f7650,0xe7b8,0x4747,0xb6ea,0x46,0xe1,0x09,0xf5,0x99,0x97);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IStore::iid()) {
      return (void *) static_cast<Component::IStore*>(this);
    }
    else return nullptr; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  /* IStore */

  /** 
   * Append data (copy-based)
   * 
   * @param key 
   * @param metadata
   * @param data 
   * @param data_len 
   * @param queue_id 
   * 
   * @return 
   */
  virtual status_t put(std::string key,
                       std::string metadata,
                       void * data,
                       size_t data_len,
                       int queue_id = 0) override;

  /** 
   * Append data (zero-copy)
   * 
   * @param key 
   * @param metadata 
   * @param data 
   * @param offset 
   * @param data_len 
   * @param queue_id 
   * 
   * @return 
   */
  virtual status_t put(std::string key,
                       std::string metadata,
                       Component::io_buffer_t data,
                       size_t offset,
                       size_t data_len,
                       int queue_id = 0) override;

  /** 
   * Get number of records
   * 
   * 
   * @return Number of records
   */
  virtual size_t get_record_count() override;

  /** 
   * Get a record by rowid
   * 
   * @param rowid Row id counting from 1
   * @param iob IO buffer
   * @param offset IO buffer offset in bytes
   * @param queue_id [optional] Queue identifier
   * 
   * @return S_OK on success
   */
  virtual status_t get(uint64_t rowid,
                       Component::io_buffer_t iob,
                       size_t offset,
                       int queue_id = 0) override;

  /** 
   * Get a record by ID (key)
   * 
   * @param key Unique key
   * @param iob IO buffer
   * @param offset IO buffer offset in bytes
   * @param queue_id [optional] Queue identifier
   * 
   * @return S_OK on success
   */
  virtual status_t get(const std::string key,
                       Component::io_buffer_t iob,
                       size_t offset,
                       int queue_id = 0) override;


  /** 
   * Get metadata for a record
   * 
   * @param rowid Row identifier counting from 1
   * 
   * @return String form of metadata
   */
  virtual std::string get_metadata(uint64_t rowid) override;

  /** 
   * Open a sequential record iterator. Requires database access.
   * 
   * @param rowid_start Start row
   * @param rowid_end End row
   * @param flags Flags
   * 
   * @return Iterator
   */
  virtual iterator_t open_iterator(uint64_t rowid_start,
                                   uint64_t rowid_end,
                                   unsigned long flags = 0) override;

  /** 
   * 
   * 
   * @param expr 
   * @param prefetch_buffers 
   * 
   * @return 
   */
  virtual iterator_t open_iterator(std::string expr,
                                   unsigned long flags = 0) override;

  /** 
   * Get record count for an iterator
   * 
   * @param iter Iterator
   * 
   * @return Number of records
   */
  virtual size_t iterator_record_count(iterator_t iter) override;

  /** 
   * Get the data size for all data under an iterator
   * 
   * @param iter Iterator
   * 
   * @return Size in bytes
   */
  virtual size_t iterator_data_size(iterator_t iter) override;
  
  /** 
   * Get data size for next record in iterator
   * 
   * @param iter Iterator
   * 
   * @return Size of record in bytes
   */
  virtual size_t iterator_next_record_size(iterator_t iter) override;

  /** 
   * Close iterator
   * 
   * @param iter Iterator
   */
  virtual void close_iterator(iterator_t iter) override;


  /** 
   * Read from an iterator.  Does not require database access.
   * 
   * @param iter Iterator
   * @param iob IO buffer
   * @param offset Offset in IO buffer
   * @param queue_id [optional] Queue identifier
   * 
   * @return Number of bytes transferred
   */
  virtual size_t iterator_get(iterator_t iter,
                              Component::io_buffer_t* iob,
                              size_t offset,
                              int queue_id = 0) override;

  /** 
   * Read from an iterator.  Does not require database access.
   * 
   * @param iter Iterator
   * @param iob [out] IO buffer
   * @param queue_id [optional] Queue identifier
   * 
   * @return Number of bytes transferred
   */
  virtual size_t iterator_get(iterator_t iter,
                              Component::io_buffer_t& iob,
                              int queue_id = 0) override;

  /** 
   * Split iterator into multiple iterators (which can be passed to
   * separate threads)
   * 
   * @param iter Iterator
   * @param ways Number of ways to split
   */
  virtual void split_iterator(iterator_t iter,
                              size_t ways,
                              std::vector<iterator_t>& out_iter_vector) override;

  /** 
   * Reset an interator to initial index
   * 
   * @param iter Iterator
   */
  virtual void reset_iterator(iterator_t iter) override;

  /** 
   * Get metadata from filter expression
   * 
   * @param filter_expr Filter expression
   * @param out_metadata [out] metadata
   *
   * @return Row id
   */
  virtual size_t fetch_metadata(const std::string filter_expr,
                                std::vector<std::pair<std::string,std::string> >& out_metadata) override;
  
  /** 
   * Determine if a path is valid.
   * 
   * @param path Path
   * 
   * @return Row id
   */
  virtual uint64_t check_path(const std::string path) override;

  /** 
   * Dump debugging information
   * 
   */
  virtual void dump_info() override;

  /** 
   * Flush queued IO and wait for completion
   * 
   * 
   * @return S_OK on success
   */
  virtual status_t flush() override;

  inline Core::Physical_memory * phys_mem_allocator() {
    return &_phys_mem_allocator;
  }
  
private:
  void show_db();
  void execute_sql(const std::string& sql, bool print_callback = false);
  uint64_t insert_row(std::string& key,
                      std::string& metadata,
                      uint64_t lba,
                      uint64_t length);
  bool find_row(std::string& key, uint64_t& out_lba);

  void monitor_thread_entry();

  sqlite3 * db_handle();
  
private:
  /* component dependencies and instantiations */
  Component::IBlock_device * _block;
  Core::Physical_memory      _phys_mem_allocator;
  Header                     _hdr;
  Component::VOLUME_INFO     _vi;
  unsigned                   _max_io_blocks;
  unsigned                   _max_io_bytes;
  std::string                _db_filename;
  std::string                _table_name;
  bool                       _read_only;
  
  std::thread                _monitor;
  bool                       _monitor_exit = false;

  /* stats collection */
  struct {
    uint64_t iterator_get_volume;
  } stats __attribute__((aligned(8)));
};


class Append_store_factory : public Component::IStore_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfacf7650,0xe7b8,0x4747,0xb6ea,0x46,0xe1,0x09,0xf5,0x99,0x97);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IStore_factory::iid()) {
      return (void *) static_cast<Component::IStore_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IStore * create(const std::string owner,
                                     const std::string name,
                                     const std::string db_location,
                                     Component::IBlock_device* block,
                                     int flags) override
  {
    if(block == nullptr)
      throw Constructor_exception("%s: bad block interface param", __PRETTY_FUNCTION__);
    
    Component::IStore * obj = static_cast<Component::IStore *>
      (new Append_store(owner, name, db_location, block, flags));
    
    obj->add_ref();
    return obj;
  }

};



#endif // __APPEND_STORE_H__
