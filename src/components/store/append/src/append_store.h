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
#ifndef __APPEND_STORE_H__
#define __APPEND_STORE_H__

#include <sqlite3.h>
#include <string>
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

public:

  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  Append_store(std::string owner,
               std::string name,
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
   * Get record count for an iterator
   * 
   * @param iter Iterator
   * 
   * @return Number of records
   */
  virtual size_t record_count(iterator_t iter) override;
  
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

private:
  /* component dependencies and instantiations */
  Component::IBlock_device * _block;
  Core::Physical_memory      _phys_mem_allocator;
  Header                     _hdr;
  Component::VOLUME_INFO     _vi;
  unsigned                   _max_io_blocks;
  unsigned                   _max_io_bytes;
  sqlite3 *   _db;
  std::string _db_filename;
  std::string _table_name; 
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

  virtual Component::IStore * create(std::string owner,
                                     std::string name,
                                     Component::IBlock_device* block,
                                     int flags) override
  {
    if(block == nullptr)
      throw Constructor_exception("%s: bad block interface param", __PRETTY_FUNCTION__);
    
    Component::IStore * obj = static_cast<Component::IStore *>
      (new Append_store(owner, name, block, flags));
    
    obj->add_ref();
    return obj;
  }

};



#endif // __APPEND_STORE_H__
