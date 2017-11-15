#pragma once

#ifndef __BLOB_BIG_COMPONENT_H__
#define __BLOB_BIG_COMPONENT_H__

#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/pager_itf.h>
#include <api/blob_itf.h>

#include <core/avl_malloc.h>
#include <mutex>
#include <string>
#include <list>
#include "data_region.h"

class Blob_component_factory : public Component::IBlob_factory
{
public:
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfacd40ac,0x5a61,0x4489,0xae09,0xcf,0x7e,0x29,0x8b,0xb9,0x90);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IBlob_factory::iid()) {
      return (void *) static_cast<Component::IBlob_factory*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  /** 
   * Open a Blob store
   * 
   * @param owner Owner identifier
   * @param name Store name
   * @param block_device Underlying block device
   * @param flags Instantiation flags
   * 
   * @return Pointer to IBlob interface
   */
  virtual Component::IBlob * open(std::string owner,
                                  std::string name,
                                  Component::IRegion_manager * rm,
                                  size_t value_space_mb,
                                  int flags) override;

};


class Blob_component : public Component::IBlob
{  
private:
  static constexpr bool option_DEBUG = true;
  static constexpr size_t NUM_BLOCKS_FOR_METADATA = 5;
  static constexpr size_t BLOCK_SIZE = 4096;
    
public:
  Blob_component(std::string owner,
                 std::string name,
                 Component::IRegion_manager * region_manager,
                 size_t value_space_mb,
                 int flags);
  
  /** 
   * Destructor
   * 
   */
  virtual ~Blob_component();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x320d40ac,0x5a61,0x4489,0xae09,0xcf,0x7e,0x29,0x8b,0xb9,0x90);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IBlob::iid()) {
      return (void *) static_cast<Component::IBlob*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }
  
  virtual status_t start() override { return E_NOT_IMPL; } 
  virtual status_t stop() override { return E_NOT_IMPL; } 
  virtual status_t shutdown() override { return E_NOT_IMPL; } 
  virtual status_t reset() override { return E_NOT_IMPL; } 

public:
  /** 
   * IBlob
   */  

  /** 
   * Create a new blob
   * 
   * @param size_in_bytes Initial size of blob
   * 
   * @return Handle to new blob
   */
  virtual blob_t create(size_t size_in_bytes) override;

  /** 
   * Erase a blob
   * 
   * @param handle Blob handle
   */
  virtual void erase(blob_t handle) override;

  /** 
   * Open a cursor to a blob
   * 
   * @param handle Blob handle
   * 
   * @return Cursor
   */
  virtual cursor_t open(blob_t handle) override;

  /** 
   * Close a cursor to a blob
   * 
   * @param handle 
   * 
   * @return 
   */
  virtual cursor_t close(blob_t handle) override;

  /** 
   * Move cursor position
   * 
   * @param cursor Cursor handle
   * @param offset Offset in bytes
   * @param flags SEEK_SET etc.
   */
  virtual void seek(cursor_t cursor, long offset, int flags) override;

  /** 
   * Zero copy version of read
   * 
   * @param cursor 
   * @param buffer 
   * @param buffer_offset 
   * @param size_in_bytes 
   */
  virtual void read(cursor_t cursor, Component::io_buffer_t buffer, size_t buffer_offset, size_t size_in_bytes) override;

  /** 
   * Copy-based read
   * 
   * @param cursor 
   * @param dest 
   * @param size_in_bytes 
   */
  virtual void read(cursor_t cursor, void * dest, size_t size_in_bytes) override;

  /** 
   * Zero copy version of write
   * 
   * @param cursor 
   * @param buffer 
   * @param buffer_offset 
   * @param size_in_bytes 
   */
  virtual void write(cursor_t cursor,
                     Component::io_buffer_t buffer,
                     size_t buffer_offset,
                     size_t size_in_bytes) override;

  /** 
   * Copy-based write
   * 
   * @param cursor 
   * @param dest 
   * @param size_in_bytes 
   */
  virtual void write(cursor_t cursor,
                     void * dest,
                     size_t size_in_bytes) override;


  /** 
   * Set the size of the file (like POSIX truncate call)
   * 
   * @param size_in_bytes Size in bytes
   */
  virtual void truncate(blob_t handle, size_t size_in_bytes) override;


  /** 
   * Debug state of the blob store
   * 
   * @param filter 
   */
  virtual void show_state(std::string filter) override;
  
private:
  void flush_md();
  
private:
  Component::IRegion_manager *                 _rm;
  
  Component::IBlock_device *                   _block_md; /*< block device for metadata */
  Component::VOLUME_INFO                       _block_md_vi;
  Component::io_buffer_t                       _block_md_iob;
  Core::Slab::Allocator<Blob::Data_region> *   _slab;
  std::mutex                                   _md_lock;

  Component::IBlock_device *                   _block_data; /*< block device for data */
  Component::VOLUME_INFO                       _block_data_vi;
  Blob::Data_region_tree<Blob::Data_region> *  _data_allocator;

};

#endif
