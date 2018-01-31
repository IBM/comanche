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

#pragma once

#ifndef __API_BLOB_ITF__
#define __API_BLOB_ITF__

#include <api/memory_itf.h>
#include <api/region_itf.h>

namespace Component
{

/** 
 * Blob store interface that permits chains of blocks that make up a blob.
 */
class IBlob : public Component::IBase
{
public:
  
  DECLARE_INTERFACE_UUID(0xb114511d,0x991c,0x4ca9,0xb8b7,0x79,0x09,0x15,0xd5,0xab,0x6b);

public:
  using blob_t = void*;
  using cursor_t = void*;

  /** 
   * Create a new blob
   * 
   * @param name Name of blob
   * @param owner Optional owner identifier
   * @param datatype Optional data type
   * @param size_in_bytes Initial size of blob in bytes
   * 
   * @return Handle to new blob
   */
  virtual blob_t create(const std::string& name,
                        const std::string& owner,
                        const std::string& datatype,
                        size_t size_in_bytes) = 0;

  /** 
   * Erase a blob
   * 
   * @param handle Blob handle
   */
  virtual void erase(blob_t handle) = 0;

  /** 
   * Open a cursor to a blob
   * 
   * @param handle Blob handle
   * 
   * @return Cursor
   */
  virtual cursor_t open(blob_t handle) = 0;

  /** 
   * Close a cursor to a blob
   * 
   * @param handle 
   * 
   * @return 
   */
  virtual cursor_t close(blob_t handle) = 0;

  /** 
   * Move cursor position
   * 
   * @param cursor Cursor handle
   * @param offset Offset in bytes
   * @param flags SEEK_SET etc.
   */
  virtual void seek(cursor_t cursor, long offset, int flags) = 0;

  /** 
   * Zero copy version of read
   * 
   * @param cursor 
   * @param buffer 
   * @param buffer_offset 
   * @param size_in_bytes 
   */
  virtual void read(cursor_t cursor, io_buffer_t buffer, size_t buffer_offset, size_t size_in_bytes) = 0;

  /** 
   * Copy-based read
   * 
   * @param cursor 
   * @param dest 
   * @param size_in_bytes 
   */
  virtual void read(cursor_t cursor, void * dest, size_t size_in_bytes) = 0;

  /** 
   * Zero copy version of write
   * 
   * @param cursor 
   * @param buffer 
   * @param buffer_offset 
   * @param size_in_bytes 
   */
  virtual void write(cursor_t cursor, io_buffer_t buffer, size_t buffer_offset, size_t size_in_bytes) = 0;

  /** 
   * Copy-based write
   * 
   * @param cursor 
   * @param dest 
   * @param size_in_bytes 
   */
  virtual void write(cursor_t cursor, void * dest, size_t size_in_bytes) = 0;

  /** 
   * Set the size of the file (like POSIX truncate call)
   * 
   * @param size_in_bytes Size in bytes
   */
  virtual void truncate(blob_t handle, size_t size_in_bytes) = 0;

  /** 
   * Debug state of the blob store
   * 
   * @param filter 
   */
  virtual void show_state(std::string filter) = 0;
};


class IBlob_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfacb1848,0x321a,0x4f85,0x898b,0xff,0xdf,0x12,0x4a,0x23,0x70);


  enum {
    FLAGS_CREATE = 0x1,
    FLAGS_FORMAT = 0x2, /*< force region manager to format block device */
  };

  /** 
   * Open a blob store from a block device
   * 
   * @param owner Owner identifier
   * @param name Store name
   * @param base_block_device Underlying block device
   * @param flags Instantiation flags
   * 
   * @return Pointer to IRange_manager interface
   */
  virtual IBlob * open(std::string owner,
                       std::string name,
                       Component::IBlock_device * base_block_device,
                       int flags) = 0;

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
                                  int flags) = 0;

};


}


#endif 
