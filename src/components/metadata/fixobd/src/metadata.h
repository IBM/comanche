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

#ifndef __METADATA_COMPONENT_H__
#define __METADATA_COMPONENT_H__

#include <api/metadata_itf.h>
#include "md_record.h"

struct __md_record;


class Metadata : public Component::IMetadata
{  
private:
  static constexpr bool option_DEBUG = true;
  
public:
  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  Metadata(Component::IBlock_device * block_device,
           unsigned block_size,
           int flags);


  /** 
   * Destructor
   * 
   */
  virtual ~Metadata();

  inline void lock(unsigned long index) {
    _lock_array->lock(index);
  }

  inline void unlock(unsigned long index) {
    _lock_array->unlock(index);
  }

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xb2220906,0x1eec,0x48ec,0xa643,0x29,0xeb,0x6c,0x06,0x70,0xd2);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IMetadata::iid()) {
      return (void *) static_cast<Component::IMetadata*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  /* IMetadata */
  
  /** 
   * Get total number of records
   * 
   * 
   * @return Total number of records
   */
  virtual size_t get_record_count() override;

  /** 
   * Open iterator
   * 
   * @param filter Filter string (e.g., JSON) 
   * 
   * @return Handle to iterator
   */
  virtual iterator_t open_iterator(std::string filter) override;

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
                                uint64_t* lba_count = nullptr) override;

  /** 
   * Get number of records in an iterator
   * 
   * @param iter Iterator handle
   * 
   * @return Number of records
   */
  virtual size_t iterator_record_count(iterator_t iter) override;

  /** 
   * Close an iterator and free memory
   * 
   * @param iterator Iterator handle
   */
  virtual void close_iterator(iterator_t iterator) override;

    /** 
   * Allocate a free metadata entry
   * 
   */
  virtual void allocate(uint64_t start_lba,
                        uint64_t lba_count,
                        const char * id,
                        const char * owner,
                        const char * datatype) override;

  /** 
   * Show metadata state
   * 
   */
  void dump_info() override;

private:
  
  /** 
   * Check record integrity (without holding any locks)
   * 
   * 
   * @return True on OK
   */
  bool scan_records_unsafe();

  /** 
   * Wipe and reinitialize 
   * 
   */
  void reinitialize();


  /** 
   * Apply function to each record
   * 
   */
  void apply(std::function<void(struct __md_record&, size_t index, bool& keep_going)>,
             unsigned long start,
             unsigned long count);

  /** 
   * Flush metadata record to storage
   * 
   * @param record 
   */
  void flush_record(struct __md_record* record);

  
private:
  Component::IBlock_device * _block;
  size_t                     _n_records; /*< number of records */
  size_t                     _memory_size;
  Component::VOLUME_INFO     _vi;
  size_t                     _block_size; /*< may different than underlying block size */
  Component::io_buffer_t     _iob;
  struct __md_record *       _records;
  Lock_array *               _lock_array;
};


class Metadata_factory : public Component::IMetadata_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac20906,0x1eec,0x48ec,0xa643,0x29,0xeb,0x6c,0x06,0x70,0xd2);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IMetadata_factory::iid()) {
      return (void *) static_cast<Component::IMetadata_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IMetadata * create(Component::IBlock_device * block_device,
                                        unsigned block_size,
                                        int flags) override
  {
    Component::IMetadata * obj = static_cast<Component::IMetadata*>
      (new Metadata(block_device, block_size, flags));
    
    obj->add_ref();
    return obj;
  }

};



#endif
