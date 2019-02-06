/*
   Copyright [2019] [IBM Corporation]

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

#ifndef __NUPM_DAX_MAP_H__
#define __NUPM_DAX_MAP_H__

#include <string>
#include <mutex>
#include "nd_utils.h"

namespace nupm
{

class DM_region_header;

/** 
 * Lowest level persisent manager for devdax devices. See dax_map.cc for static configuration.
 * 
 */
class Devdax_manager
{
private:
  static constexpr unsigned _debug_level = 3;
  
public:
  /** 
   * Constructor
   * 
   * @param force_reset If true, contents will be re-initialized. If false, re-initialization occurs on bad version/magic detection.
   */
  Devdax_manager(bool force_reset = false);

  /** 
   * Destructor will not unmap memory/nor invalidate pointers?
   * 
   */
  ~Devdax_manager();

  /** 
   * Open a region of memory
   * 
   * @param uuid Unique identifier
   * @param numa_node NUMA node
   * @param out_length Out length of region in bytes
   * 
   * @return Pointer to mapped memory or nullptr on not found
   */
  void * open_region(uint64_t uuid, int numa_node, size_t * out_length);

  /** 
   * Create a new region of memory
   * 
   * @param uuid Unique identifier
   * @param numa_node NUMA node
   * @param size Size of the region requested in bytes
   * 
   * @return Pointer to mapped memory
   */
  void * create_region(uint64_t uuid, int numa_node, size_t size);

  /** 
   * Erase a previously allocated region
   * 
   * @param uuid Unique region identifier
   * @param numa_node NUMA node
   */
  void erase_region(uint64_t uuid, int numa_node);

  /** 
   * Get the maximum "hole" size.
   * 
   * 
   * @return Size in bytes of max hole
   */
  size_t get_max_available(int numa_node);
  
  /** 
   * Debugging information
   * 
   * @param numa_node 
   */
  void debug_dump(int numa_node);
  
private:
  void * get_devdax_region(const char * device_path, size_t * out_length);
  void * map_region(const char * path, addr_t base_addr);
  void recover_metadata(const char * device_path, void * p, size_t p_len, bool force_rebuild = false);
  
private:
  using guard_t = std::lock_guard<std::mutex>;

  /* singleton pattern */
  static ND_control                                _nd;
  static std::map<std::string, iovec>              _mapped_regions;
  static std::map<std::string, DM_region_header *> _region_hdrs;
  static std::mutex                                _reentrant_lock;

};


}

#endif
