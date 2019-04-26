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


/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __API_BLOB_ITF__
#define __API_BLOB_ITF__

#include <stdint.h>
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
  using blob_t = uint64_t;
  using cursor_t = uint64_t;

  /** 
   * Create a new blob and initialize to zero.
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
   * Open cursor to blob
   * 
   * @param blob Blob handle
   * 
   * @return Cursor handle
   */
  virtual cursor_t open_cursor(blob_t blob) = 0;
  
  /** 
   * Synchronous direct read into IO buffer
   * 
   * @param cursor Cursor handle
   * @param iob IO buffer handle
   * @param size_in_bytes Number of bytes to read
   * @param offset Offset in bytes
   * 
   * @return S_OK on success
   */
  virtual status_t read(cursor_t cursor,
                        Component::io_buffer_t& iob,
                        size_t size_in_bytes,
                        size_t iob_offset = 0) = 0;


  /** 
   * Debug state of the blob store
   * 
   * @param filter 
   */
  virtual void show_state(std::string filter) = 0;

  /** 
   * Check if blob exists
   * 
   * @param key Name of blob
   * @param out_size [output] Size of blob in bytes
   * 
   * @return True if blob exists
   */
  virtual bool check_key(const std::string& key, size_t& out_size) = 0;

  /** 
   * Get a vector of metadata 
   * 
   * @param filter Filter expression
   * @param out_vector 
   *
   */
  virtual void get_metadata_vector(const std::string& filter,
                                   std::vector<std::string>& out_vector) = 0;
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
