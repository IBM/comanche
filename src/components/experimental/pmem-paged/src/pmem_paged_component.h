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
#pragma once

#ifndef __PMEM_PAGED_COMPONENT_H__
#define __PMEM_PAGED_COMPONENT_H__

#include <api/block_itf.h>
#include <api/pmem_itf.h>
#include <api/pager_itf.h>

#include <string>
#include <list>
#include <thread>

class Pmem_paged_component : public Component::IPersistent_memory
{  
private:
  static constexpr bool option_DEBUG = false;
  
public:
  Pmem_paged_component(std::string heap_set_id, Component::IPager * pager);
  
  /** 
   * Destructor
   * 
   */
  virtual ~Pmem_paged_component();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xeb6e37b3,0xff33,0x484b,0x8da1,0x68,0x8f,0x9f,0x60,0xaf,0xbd);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IPersistent_memory::iid()) {
      return (void *) static_cast<Component::IPersistent_memory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  /* IPersistent_memory */

  
  /** 
   * Allocate a region of persistent memory (block device backed)
   * 
   * @param id Unique region identifier
   * @param size Size of virtual space in bytes
   * @param vptr [out] virtual address pointer
   * 
   * @return Handle
   */
  virtual pmem_t allocate(std::string id,
                          size_t size,
                          void** vptr) override;

  /** 
   * Free a previously allocated persistent memory region
   * 
   * @param handle Handle to persistent memory region
   */
  virtual void free(pmem_t handle) override;

  /** 
   * Determine if a virtual address is in persistent memory
   * 
   * @param p Virtual address pointer
   * 
   * @return True if belonging to persistent memory
   */
  virtual bool is_pmem(void * p) override;

  /** 
   * Flush all volatile data to peristent memory in a non-transacation context.
   * 
   */
  virtual void persist() override;

  /** 
   * Start transaction
   * 
   */
  virtual void tx_begin() override;

  /** 
   * Commit transaction
   * 
   * 
   * @return S_OK or E_FAIL
   */
  virtual status_t tx_commit() override;

  virtual size_t fault_count() override { return 0; } // TODO
  
private:

  status_t start_thread();
  void pf_thread_entry();
  
private:

  enum {
    THREAD_UNINIT = 0,
    THREAD_RUNNING = 1,
    THREAD_EXIT = 2,
  };
  
  std::thread *              _pf_thread;
  int                        _pf_thread_status;
  bool                       _exit_thread = false;
  Component::IPager *        _pager;
  Component::VOLUME_INFO     _vi;
  int                        _fdmod;

};


class Pmem_paged_component_factory : public Component::IPersistent_memory_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xface37b3,0xff33,0x484b,0x8da1,0x68,0x8f,0x9f,0x60,0xaf,0xbd);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IPersistent_memory_factory::iid()) {
      return (void *) static_cast<Component::IPersistent_memory_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IPersistent_memory * open_heap_set(std::string heap_set_id, Component::IPager * pager) override
  {
    using namespace Component;
    IPersistent_memory * pm = static_cast<IPersistent_memory*>(new Pmem_paged_component(heap_set_id, pager));
    pm->add_ref();
    return pm;
  }
};



#endif
