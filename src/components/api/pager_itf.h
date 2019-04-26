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

#ifndef __API_PAGER_ITF_H__
#define __API_PAGER_ITF_H__

#include <string>
#include <component/base.h>
#include "api/block_itf.h"

namespace Component
{

/** 
 * Interface definition for IPager
 *
 * 
 */
class IPager : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xc7ac2b3c,0x7989,0x11e3,0x8f16,0xbc,0x30,0x5b,0xdc,0x07,0x17);
  
  /** 
   * Get a paged region
   * 
   * @param id Identifier of region (persistent info)
   * @param size Size of region in bytes
   *
   * @return Virtual load address
   */
  virtual void * get_region(std::string id, size_t size, bool& reused) = 0; 

  /** 
   * 
   * Request page mapping/eviction pair
   *
   * @param virt_addr_faulted virtual address which trigger the faults
   * @param virt_addr_pinned, virt addr to access the pinned physical page(since the faulting address is no mapped yet)
   * @param p_phys_addr_faulted physical addr to map to faulted address
   * @param p_virt_addr_evicted 
   * @param is_young young means has been access before(swapped out previously)
   * 
   */
  virtual void request_page(addr_t virt_addr_faulted,
                            addr_t *p_phys_addr_faulted,
                            addr_t *p_virt_addr_evicted) = 0;

  /** 
   * Clear mappings for a given virtual address range. Flush out anything held in memory.
   * 
   * @param vaddr Virtual address of the region
   * @param size Size in bytes of the region
   */
  virtual void clear_mappings(addr_t vaddr, size_t size) = 0;

  /** 
   * Flush data to store
   * 
   */
  virtual void flush(addr_t vaddr, size_t size) = 0;

  /** 
   * Get the write-atomicity size for the underlying device.
   * 
   * 
   * @return Atomicity size in bytes (usually 4096)
   */
  virtual size_t get_atomicity_size() = 0;
};

class IPager_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfacc2b3c,0x7989,0x11e3,0x8f16,0xbc,0x30,0x5b,0xdc,0x07,0x17);

  /** 
   * Create pager component instance. Ref count will be incremented implicitly.
   * 
   * @param nr_pages Number of physical memory pages
   * @param heap_set_id Heap set identifier
   * @param block_device Block device
   */
  virtual IPager * create(size_t nr_pages,
                          std::string heap_set_id,
                          Component::IBlock_device * block_device,
                          bool force_init = false) = 0;

};


}// Component

#endif // __API_PAGER_ITF_H__
