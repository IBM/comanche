/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */
#ifndef __API_METADATA_ITF__
#define __API_METADATA_ITF__

#include <common/types.h>
#include <api/components.h>
#include <api/block_itf.h>

namespace Component
{

/** 
 * IMetadata - metadata interface
 */
class IMetadata : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xd6fb362d,0xd67c,0x4393,0xae6d,0x7d,0xf7,0x5a,0x3a,0x7e,0xfe);

public:
  using iterator_t = void*;
  using index_t = int64_t;
  
  /** 
   * Get total number of records
   * 
   * 
   * @return Total number of used records
   */
  virtual size_t get_record_count() = 0;

  /** 
   * Open iterator
   * 
   * @param filter Filter string (e.g., JSON) 
   * 
   * @return Handle to iterator
   */
  virtual iterator_t open_iterator(std::string filter) = 0;

  /** 
   * Get next record in iteration
   * 
   * @param iter Iterator handle
   * @param out_index Out index of the record (can be used to free etc.)
   * @param out_metadata Out metadata (e.g., JSON string)
   * @param lba [optional] Out logical block address
   * @param lba_count [optional] Out logical block count
   * 
   * @return S_OK or E_EMPTY
   */
  virtual status_t iterator_get(iterator_t iter,
                                index_t& out_index,
                                std::string& out_metadata,
                                uint64_t* lba = nullptr,
                                uint64_t* lba_count = nullptr) = 0;


  /** 
   * Close an iterator and free memory
   * 
   * @param iterator Iterator handle
   */
  virtual void close_iterator(iterator_t iterator) = 0;

  /** 
   * Allocate a free metadata entry
   * 
   * @param start_lba 
   * @param lba_count 
   * @param id 
   * @param owner 
   * @param datatype 
   * 
   * @return Index >= 0 or -E_EMPTY
   */
  virtual index_t allocate(uint64_t start_lba,
                           uint64_t lba_count,
                           const std::string& id,
                           const std::string& owner,
                           const std::string& datatype) = 0;

  /** 
   * Free/delete metadata entry
   * 
   * @param index 
   */
  virtual void free(index_t index) = 0;
  
  /** 
   * Lock a metadata entry
   * 
   * @param index Entry index
   */
  virtual void lock_entry(index_t index) = 0;

  /** 
   * Unlock a metadata entry
   * 
   * @param index Entry index
   */
  virtual void unlock_entry(index_t index) = 0;

  /** 
   * Output debugging information
   * 
   */
  virtual void dump_info() = 0;

  /** 
   * Check if metadata entry exists
   * 
   * @param id 
   * @param owner 
   * @param out_size [output] Size in bytes of entity
   * 
   * @return True if entry exists
   */
  virtual bool check_exists(const std::string& id, const std::string& owner, size_t& out_size) = 0;

  /** 
   * Get string form of metadata for a given index
   * 
   * @param index Entry index
   * 
   * @return String version of metadata
   */
  virtual std::string get_metadata(index_t index) = 0;
};



class IMetadata_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfacb362d,0xd67c,0x4393,0xae6d,0x7d,0xf7,0x5a,0x3a,0x7e,0xfe);

  /** 
   * Create blob store from a block device
   * 
   * @param block_device Underlying block device
   * @param flags Instantiation flags
   * 
   * @return Pointer to IStore interface
   */  
  virtual IMetadata * create(Component::IBlock_device * block_device,
                             unsigned block_size,
                             int flags) = 0;

};


}

#endif 
