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


#ifndef __API_STORE_ITF__
#define __API_STORE_ITF__

#include <api/components.h>
#include <api/block_itf.h>

namespace Component
{

/** 
 * Blob store interface that permits chains of blocks that make up a blob.
 */
class IStore : public Component::IZerocopy_memory
{
public:
  DECLARE_INTERFACE_UUID(0x4c3e4f58,0x4ade,0x4994,0x9a04,0x0c,0xd2,0x9a,0x9b,0x85,0x55);

public:
  using iterator_t=void*;

  /** 
   * Put-append data (copy-based)
   * 
   * @param key 
   * @param metadata 
   * @param data 
   * @param data_len 
   * @param queue_id [optional] Queue identifier
   * 
   * @return 
   */
  virtual status_t put(std::string key,
                       std::string metadata,
                       void * data,
                       size_t data_len,
                       int queue_id = 0) = 0;

  /** 
   * Zero-copy put-append
   * 
   * @param key 
   * @param metadata 
   * @param data IO buffer holding data
   * @param offset Offset in IO buffer
   * @param data_len 
   * @param queue_id [optional] Queue identifier
   * 
   * @return 
   */
  virtual status_t put(std::string key,
                       std::string metadata,
                       Component::io_buffer_t data,
                       size_t offset,
                       size_t data_len,
                       int queue_id = 0) = 0;

  /** 
   * Get number of records
   * 
   * 
   * @return Number of records
   */
  virtual size_t get_record_count() = 0;


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
                   int queue_id = 0) = 0;

  /** 
   * Get metadata for a record
   * 
   * @param rowid Row identifier counting from 1
   * 
   * @return String form of metadata
   */
  virtual std::string get_metadata(uint64_t rowid) = 0;
  
  /** 
   * Open a sequential record iterator. Requires database access.
   * 
   * @param rowid_start Start row
   * @param rowid_end End row
   * @param prefetch_buffers Number of prefetch buffers
   * 
   * @return Iterator
   */
  virtual iterator_t open_iterator(uint64_t rowid_start,
                                   uint64_t rowid_end,
                                   unsigned prefetch_buffers = 0) = 0;

  /** 
   * Close iterator
   * 
   * @param iter Iterator
   */
  virtual void close_iterator(iterator_t iter) = 0;


  /** 
   * Read from an iterator.  Does not require database access. Use callee allocated iob.
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
                              int queue_id = 0) = 0;

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
                              int queue_id = 0) = 0;  

  /** 
   * Free buffer previously returned from iterator_get method
   * 
   * @param iter Iterator
   * @param iob IO buffer
   */
  virtual void free_iterator_buffer(iterator_t iter, Component::io_buffer_t iob) = 0;
  

  /** 
   * Dump debugging information
   * 
   */
  virtual void dump_info() { }
  
  /** 
   * Flush queued IO and wait for completion
   * 
   * 
   * @return S_OK on success
   */
  virtual status_t flush() = 0;
};


class IStore_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xface4f58,0x4ade,0x4994,0x9a04,0x0c,0xd2,0x9a,0x9b,0x85,0x55);

  /** 
   * Create blob store from a block device
   * 
   * @param owner Owner identifier
   * @param name Store name
   * @param region_device Underlying region manager
   * @param value_space_mb Size of value-space in MB
   * @param flags Instantiation flags
   * 
   * @return Pointer to IRange_manager interface
   */
  virtual IStore * create(std::string owner,
                          std::string name,
                          Component::IBlock_device * block_device,
                          int flags) = 0;

};


}


#endif 
