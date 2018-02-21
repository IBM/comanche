#ifndef __BLOB_BIG_COMPONENT_H__
#define __BLOB_BIG_COMPONENT_H__

#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/pager_itf.h>
#include <api/blob_itf.h>
#include <api/metadata_itf.h>
#include <api/block_allocator_itf.h>

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
                                  Component::IBlock_device * base_block_device,
                                  int flags) override;

  /** 
   * Late binding open
   * 
   * @param owner Owner
   * @param name Store name
   * @param flags Instantiation flags
   * 
   * @return Pointer to IBlock interface
   */  
  virtual Component::IBlob * open(std::string owner,
                                  std::string name,
                                  int flags) override;

};


class Blob_component : public Component::IBlob
{  
private:
  static constexpr bool option_DEBUG = true;
  
  static constexpr size_t BLOCK_SIZE = 4096;
  static constexpr size_t ALLOCATOR_SIZE_BYTES = MB(8);
  static constexpr size_t METADATA_SIZE_BYTES =  MB(8);
  
public:
  Blob_component(std::string owner,
                 std::string name,
                 Component::IBlock_device * base_block_device,
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

  virtual int bind(Component::IBase * base);
  virtual int release_bindings() { throw API_exception("cannot release bindings"); return 0; }

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
   * @param name Name of blob
   * @param owner Owner
   * @param datatype Optional data type
   * @param size_in_bytes Initial size of blob in bytes
   * 
   * @return Handle to new blob
   */
  virtual blob_t create(const std::string& name,
                        const std::string& owner,
                        const std::string& datatype,
                        size_t size_in_bytes) override;

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

  /** 
   * Check if blob exists
   * 
   * @param key Name of blob
   * @param out_size [output] Size of blob in bytes
   * 
   * @return True if blob exists
   */
  virtual bool check_key(const std::string& key, size_t& out_size) override;

  /** 
   * Get a vector of metadata 
   * 
   * @param name_filter
   * @param out_vector 
   */
  virtual void get_metadata_vector(const std::string& filter,
                                   std::vector<std::string>& out_vector) override;


  
private:
  void instantiate_components();
  void flush_md();
  
private:
  const std::string                            _owner;
  const std::string                            _name;
  int                                          _flags;
  Component::IBlock_device *                   _base_block_device; /*< base block device is split into regions for allocator, metadata and data */
  Component::VOLUME_INFO                       _base_block_device_vi;
  Component::IRegion_manager *                 _base_rm;

  /*------------------------------------------------------------------------*/
  /* split block device into three regions: block allocator, metadata, data */
  /*------------------------------------------------------------------------*/

  Component::IBlock_device *                   _block_allocator; /*< block device for allocator */
  Component::IBlock_device *                   _block_md; /*< block device for metadata */
  Component::IBlock_device *                   _block_data; /*< block data */
  Component::IPersistent_memory *              _pmem_allocator;
  Component::IBlock_allocator *                _allocator;
  Component::IMetadata *                       _metadata;

};

#endif
