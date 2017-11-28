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
   * @return Total number of records
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
   * @param out_metadata Out metadata (e.g., JSON string)
   * @param allocator_handle Out allocator handle (see allocator_itf.h)
   * @param lba [optional] Out logical block address
   * @param lba_count [optional] Out logical block count
   * 
   * @return S_OK or E_EMPTY
   */
  virtual status_t iterator_get(iterator_t iter,
                                std::string& out_metadata,
                                void *& allocator_handle,
                                uint64_t* lba = nullptr,
                                uint64_t* lba_count = nullptr) = 0;

  /** 
   * Get number of records in an iterator
   * 
   * @param iter Iterator handle
   * 
   * @return Number of records
   */
  virtual size_t iterator_record_count(iterator_t iter) = 0;

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
                           const char * id,
                           const char * owner,
                           const char * datatype) = 0;


  /** 
   * Output debugging information
   * 
   */
  virtual void dump_info() = 0;

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
