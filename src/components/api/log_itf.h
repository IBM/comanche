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

#ifndef __API_LOG_ITF__
#define __API_LOG_ITF__

#include <common/exceptions.h>
#include <api/components.h>
#include <api/block_itf.h>

namespace Component
{

/** 
 * Blob store interface that permits chains of blocks that make up a blob.
 */
class ILog : public Component::IZerocopy_memory
{
public:
  DECLARE_INTERFACE_UUID(0x7672b5bd,0x911d,0x4cf9,0xb550,0xb6,0xac,0xc0,0x12,0x78,0x0e);

public:

  /** 
   * Asynchronously write data (copy-based)
   * 
   * @param data Pointer to data
   * @param data_len Length of data in bytes
   * @param queue_id [optional] Queue index
   * 
   * @return Index of item
   */
  virtual index_t write(const void * data, const size_t data_len, unsigned queued_id = 0) = 0;

  /** 
   * Read data from a given offset
   * 
   * @param index Index of item
   * @param data IO buffer (must be atleast record_size + block_size)
   * @param queue_id [optional] Queue index
   * 
   * @return Pointer to record
   */
  virtual byte * read(const index_t index, Component::io_buffer_t iob, unsigned queue_id = 0) = 0;

  /** 
   * Read blob into a string (copy based)
   * 
   * @param index Index of item
   * 
   * @return String
   */
  virtual std::string read(const index_t index) = 0;

  /** 
   * Get next free byte address of storage
   * 
   * 
   * @return Index (byte offset)
   */
  virtual index_t get_tail() = 0;

  /** 
   * Return fixed size
   * 
   * 
   * @return 0 if not fixed size
   */
  virtual size_t fixed_size() = 0;
  
  /** 
   * Flush queued IO and wait for completion
   *
   * @param queue_id Optional queue ide
   * 
   * @return S_OK on success
   */
  virtual status_t flush(unsigned queue_id = 0) = 0;

  /** 
   * Dump debugging information
   * 
   */
  virtual void dump_info() { throw API_exception("not implemented"); }


};


class ILog_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfac2b5bd,0x911d,0x4cf9,0xb550,0xb6,0xac,0xc0,0x12,0x78,0x0e);

  /** 
   * Create log instance
   * 
   * @param owner Owner
   * @param name Name identifier
   * @param block_device Target block device
   * @param flags Creation flags
   * @param fixed_size If set, defines fixed size items (length in bytes)
   * @param crc32 If set, include crc32
   * 
   * @return Pointer to new instance
   */
  virtual ILog * create(std::string owner,
                        std::string name,
                        Component::IBlock_device * block_device,                      
                        int flags,
                        size_t fixed_size = 0,
                        bool crc32 = false) = 0;

};


}


#endif 
