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


#ifndef __API_PMEM_ITF__
#define __API_PMEM_ITF__

#include <string>
#include <api/pager_itf.h>
#include <api/region_itf.h>

namespace Component
{
using persist_id_t = std::string;

class IPersistent_memory;

class IPersistent_memory_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfac24af9,0x6613,0x42e7,0xb198,0xef,0xa6,0x5e,0x0c,0xf0,0x0f);

  /** 
   * [optional] Open allocator with pager
   * 
   * @param owner_id Owner identifier
   * @param pager Pager interface
   * @param force_init Force re-initialization
   * 
   * @return Pointer to persistent memory interface
   */
  virtual IPersistent_memory * open_allocator(std::string owner_id,
                                              Component::IPager * pager,
                                              bool force_init = false) { throw API_exception("not implemented"); }

  /** 
   * [optional] Open memory allocator with block device.
   * 
   * @param owner_id Owner identifier
   * @param block_device Block device interface
   * @param force_init Force re-initialization
   * 
   * @return Pointer to persistent memory interface
   */
  virtual IPersistent_memory * open_allocator(std::string owner_id,
                                              Component::IBlock_device * block_device,
                                              bool force_init = false) { throw API_exception("not implemented"); }

  /** 
   * [optional] Open memory allocator with region manager
   * 
   * @param owner_id Owner identifier
   * @param rm Region manager interface
   * @param force_init Force re-initialization
   * 
   * @return Pointer to persistent memory interface
   */
  virtual IPersistent_memory * open_allocator(std::string owner_id,
                                              Component::IRegion_manager * rm,
                                              bool force_init = false) { throw API_exception("not implemented"); }

};

/** 
 * Interface to persistent memory.  Component implementations could be
 * based on pmem.io or NVMe-backed DRAM.
 * 
 */
class IPersistent_memory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0x76624af9,0x6613,0x42e7,0xb198,0xef,0xa6,0x5e,0x0c,0xf0,0x0f);

public:
  using pmem_t = void*;

  enum {
    FLAGS_NOFLUSH = 0x1,
  };

  /** 
   * Open a region of persistent memory (block device backed)
   * 
   * @param id Unique region identifier
   * @param size Size of virtual space in bytes
   * @param vptr [out] virtual address pointer
   * 
   * @return Handle
   */
  virtual pmem_t open(persist_id_t id,
                      size_t size,
                      int numa_node,
                      bool& reused,
                      void*& vptr) = 0;

  /** 
   * Close a previously opened persistent memory region
   * 
   * @param handle Handle to persistent memory region
   * @param flags Flags
   */
  virtual void close(pmem_t handle, int flags = 0) = 0;

  /** 
   * Explicitly erase a region of memory
   * 
   * @param handle 
   */
  virtual void erase(pmem_t handle) = 0;

  /** 
   * Retrieve # of faults.
   * 
   * 
   * @return Number of faults.
   */
  virtual size_t fault_count() { return 0; }

  /** 
   * Determine if a virtual address is in persistent memory
   * 
   * @param p Virtual address pointer
   * 
   * @return True if belonging to persistent memory
   */
  virtual bool is_pmem(void * p) = 0;

  /** 
   * Flush all volatile data to peristent memory in a non-transacation context.
   *    
   * @param handle Persistent memory handle
   */
  virtual void persist(pmem_t handle) = 0;

  /** 
   * Flush a specific region of memory
   * 
   * @param handle Persistent memory handle
   * @param ptr Virtual address
   * @param size Size in bytes
   */
  virtual void persist_scoped(pmem_t handle, void *ptr, size_t size) = 0;

  /** 
   * Get size of persistent memory in bytes
   * 
   * 
   * @return Size in bytes
   */
  virtual size_t get_size(pmem_t handle) = 0;


  /** 
   * Get the write-atomicity size for the underlying device.
   * 
   * 
   * @return Atomicity size in bytes (usually 4096)
   */
  virtual size_t get_atomicity_size() = 0;
  
  /** 
   * Start transaction
   * 
   */
  virtual void tx_begin(pmem_t handle) = 0;

  /** 
   * Commit transaction
   * 
   * 
   * @return S_OK or E_FAIL
   */
  virtual status_t tx_commit(pmem_t handle) = 0;

};

} // namespace Component

#endif
